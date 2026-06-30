#!/usr/bin/env python3
"""
RayDPT: ray-conditioned multi-scale Dense-Prediction-Transformer decoder for
binaural-audio -> ERP radial depth.

Ported from `test_for_audio_implicit_full` (model_raydpt.py + train_fullmap.py)
into the two-file autoresearch layout. This file holds the model, the composite
"coarse-arch" objective (dense masked-MAE + coarse-layout + low-pass), and the
training / testing loops. `prepare.py` holds the fixed data + evaluation harness.

Architecture (256x512):
    spec -> U-Net8 encoder -> {e2 64x128, e3 32x64, e4 16x32}
    ray query pyramid {Q16, Q32, Q64} from a spherical RayBank
    audio<->ray  : GLOBAL cross-attn at coarse tokens (e4=512, e3=2048)
    fine detail  : DPT skips (1x1 conv of e2/e3/e4)
    ray<->ray    : LOCAL spherical window attention at 32x64 and 64x128
    DPT fusion coarse->fine -> head -> sigmoid ERP depth.

Usage:
    python train.py --mode train [options]
    python train.py --mode test  [options]
"""

import argparse
import copy
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare import (
    make_dataloader, compute_errors, load_gt_rgb, erp_grid, sh_basis_matrix,
    swap_audio_lr,
)


# ============================================================
# Constants (fixed, do not modify)
# ============================================================

TIME_BUDGET = 3600  # training wall-clock budget in seconds (1 hour)


# ============================================================
# Configuration
# ============================================================

class Cfg:
    """Simple nested namespace for configuration."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        items = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'Cfg({items})'


def make_config(args):
    """Build a config object from parsed CLI arguments."""
    cfg = Cfg(
        dataset=Cfg(
            name='soundspaces',
            dataset_dir=args.dataset_dir,
            split_ratio=[0.8, 0.1, 0.1],
            split_seed=42,
            depth_type='erp',
            images_size=[256, 512],
            max_depth=10.0,
            in_ch=args.in_ch,
            sample_rate=48000,
            log_spec=True,
            audio_window_m=10.0,
        ),
        model=Cfg(
            name='raydpt',
            ngf=64,
            dim=192,
            n_heads=8,
            ray_cross_layers=2,
            raydpt_win32=5,
            raydpt_win64=3,
            raydpt_lite=args.raydpt_lite,
            raydpt_full_decode=args.raydpt_full_decode,
            # ray-feature bank flags
            use_xyz=True,
            use_fourier_pe=True,
            fourier_bands=6,
            use_sh_pe=False,
            sh_order=4,
            use_mic_pe=False,
            head_r=0.0875,
            # composite-loss head / weights
            coarse_head_h=16,
            coarse_head_w=32,
            w_dense=1.0,
            w_coarse_layout=1.0,
            w_low=0.5,
            w_rel=0.1,
            w_silog=0.0,
        ),
        mode=Cfg(
            mode=args.mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=1e-4,
            optimizer=args.optimizer,
            validation_iter=1,
            num_threads=args.num_workers,
            checkpoints=args.checkpoint,
            flip_aug=args.flip_aug,
            experiment_name=args.experiment_name,
            eval_on=args.eval_on,
            vis_every=args.vis_every,
        ),
    )
    return cfg


def build_model_cfg(cfg):
    """Flatten the nested cfg into the attribute namespace RayDPT / RayBank expect."""
    h, w = int(cfg.dataset.images_size[0]), int(cfg.dataset.images_size[1])
    m = vars(cfg.model).copy()
    m.update(img_h=h, img_w=w,
             in_ch=int(getattr(cfg.dataset, 'in_ch', 2)),
             max_depth=float(cfg.dataset.max_depth))
    return SimpleNamespace(**m)


# ============================================================
# Per-ray feature bank (fixed ERP grid features)
# ============================================================

def _fourier_pe(dirs, bands):
    """(N,3) unit dirs -> (N, 3*2*bands) Fourier features."""
    freqs = (2.0 ** np.arange(bands)) * math.pi
    ang = dirs[:, :, None] * freqs[None, None, :]
    ang = ang.reshape(dirs.shape[0], -1)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


class RayBank:
    """Fixed ERP ray grid + assembled per-ray feature matrix (N, F)."""

    def __init__(self, cfg, device="cpu"):
        H, W = cfg.img_h, cfg.img_w
        self.H, self.W, self.N = H, W, H * W
        el, az = erp_grid(H, W)
        el_f, az_f = el.ravel(), az.ravel()
        dirs = np.stack([np.cos(el_f) * np.cos(az_f),
                         np.cos(el_f) * np.sin(az_f),
                         np.sin(el_f)], axis=1).astype(np.float32)   # (N,3)

        feats = []
        if cfg.use_xyz:
            feats.append(dirs)
        if cfg.use_fourier_pe:
            feats.append(_fourier_pe(dirs, cfg.fourier_bands).astype(np.float32))
        if cfg.use_sh_pe:
            feats.append(sh_basis_matrix(cfg.sh_order, el, az).astype(np.float32))
        if cfg.use_mic_pe:
            y = dirs[:, 1:2]
            feats.append(np.concatenate([y, -y], axis=1).astype(np.float32))

        feat = np.concatenate(feats, axis=1) if feats else np.zeros((self.N, 0), np.float32)
        self.feat = torch.from_numpy(feat).to(device)             # (N, F)
        self.feat_dim = feat.shape[1]


# ============================================================
# Model building blocks
# ============================================================

def conv_bn(ci, co, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, s, p, bias=False),
                         nn.BatchNorm2d(co), nn.GELU())


class Refine(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(conv_bn(ch, ch),
                                  nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(),
                                 nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = FFN(dim)

    def forward(self, q, kv):
        a, _ = self.attn(self.n1(q), kv, kv)
        q = q + a
        return q + self.ffn(self.n2(q))


class Down(nn.Module):
    """pix2pix encoder block: Conv(4,2,1) (/2) + optional BN + LeakyReLU."""

    def __init__(self, ci, co, norm=True):
        super().__init__()
        layers = [nn.Conv2d(ci, co, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(co))
        layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet8Encoder(nn.Module):
    """Explicit pix2pix-style 8-down encoder (256x512 -> 1x2). Exposes e2/e3/e4
    feature maps used as DPT tokens / skips."""
    def __init__(self, in_ch, ngf=64):
        super().__init__()
        self.e1 = Down(in_ch,   ngf,     norm=False)   # 128x256
        self.e2 = Down(ngf,     ngf * 2)               # 64x128
        self.e3 = Down(ngf * 2, ngf * 4)               # 32x64
        self.e4 = Down(ngf * 4, ngf * 8)               # 16x32
        self.e5 = Down(ngf * 8, ngf * 8)               # 8x16
        self.e6 = Down(ngf * 8, ngf * 8)               # 4x8
        self.e7 = Down(ngf * 8, ngf * 8)               # 2x4
        self.e8 = Down(ngf * 8, ngf * 8, norm=False)   # 1x2


# ---- local spherical window attention (ray <-> ray) -------------------------
def _window_kv(t, win):
    """(B,C,H,W) -> (B,C,win*win,H,W): neighbours via circular-W / replicate-H pad."""
    pad = win // 2
    t = torch.cat([t[..., -pad:], t, t[..., :pad]], dim=-1)        # circular azimuth wrap
    t = F.pad(t, (0, 0, pad, pad), mode="replicate")               # replicate elevation (poles)
    B, C, Hp, Wp = t.shape
    H, W = Hp - 2 * pad, Wp - 2 * pad
    cols = F.unfold(t, kernel_size=win)                            # (B, C*win*win, H*W)
    return cols.view(B, C, win * win, H, W)


def _geom_bias_feats(H, W, win):
    """(H, win*win, 3): [wrapped dtheta, dphi, cos angular distance] per row/offset."""
    pad = win // 2
    el = (math.pi / 2 - (torch.arange(H).float() + 0.5) / H * math.pi)     # (H,)
    offs = [(dr, dc) for dr in range(-pad, pad + 1) for dc in range(-pad, pad + 1)]
    out = torch.zeros(H, len(offs), 3)
    dphi_u, dth_u = math.pi / H, 2 * math.pi / W
    for h in range(H):
        ei = el[h]
        for k, (dr, dc) in enumerate(offs):
            ej = el[min(max(h + dr, 0), H - 1)]
            dth = dc * dth_u
            cosang = (torch.sin(ei) * torch.sin(ej)
                      + torch.cos(ei) * torch.cos(ej) * math.cos(dth))
            out[h, k] = torch.tensor([dth, dr * dphi_u, float(cosang)])
    return out


class LocalSphericalAttention(nn.Module):
    def __init__(self, dim, heads, H, W, win=5):
        super().__init__()
        self.h, self.dh, self.win = heads, dim // heads, win
        self.scale = self.dh ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.register_buffer("geom", _geom_bias_feats(H, W, win))          # (H,K,3)
        self.bias_mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, heads))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.to_qkv(x).chunk(3, 1)
        kw = _window_kv(k, self.win).view(B, self.h, self.dh, self.win * self.win, H, W)
        vw = _window_kv(v, self.win).view(B, self.h, self.dh, self.win * self.win, H, W)
        q = q.view(B, self.h, self.dh, H, W)
        attn = torch.einsum("bndhw,bndkhw->bnkhw", q, kw) * self.scale     # (B,nh,K,H,W)
        bias = self.bias_mlp(self.geom).permute(2, 1, 0)                   # (nh,K,H)
        attn = attn + bias[None, :, :, :, None]
        attn = attn.softmax(dim=2)
        out = torch.einsum("bnkhw,bndkhw->bndhw", attn, vw).reshape(B, C, H, W)
        return x + self.proj(out)


# ---- RayDPT ------------------------------------------------------------------
class RayDPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.H, self.W = cfg.img_h, cfg.img_w
        ngf = getattr(cfg, "ngf", 64); dim = cfg.dim; heads = cfg.n_heads
        nL = getattr(cfg, "ray_cross_layers", 2)
        self.enc = UNet8Encoder(getattr(cfg, "in_ch", 2), ngf)

        def bank(h, w):
            pc = copy.copy(cfg); pc.img_h, pc.img_w = h, w
            b = RayBank(pc, device="cpu"); return b.feat, b.feat_dim
        f16, fd = bank(16, 32); f32, _ = bank(32, 64); f64, _ = bank(64, 128)
        self.register_buffer("rf16", f16); self.register_buffer("rf32", f32)
        self.register_buffer("rf64", f64)
        mk_rp = lambda: nn.Sequential(nn.Linear(fd, dim), nn.GELU(), nn.Linear(dim, dim))
        self.rp16, self.rp32, self.rp64 = mk_rp(), mk_rp(), mk_rp()
        # audio kv: e4 (512 tok), e3 (2048 tok). 64-scale reuses e4 (cheap global cue).
        self.kv_e4 = nn.Linear(ngf * 8, dim)
        self.kv_e3 = nn.Linear(ngf * 4, dim)
        mk_cr = lambda: nn.ModuleList([CrossBlock(dim, heads) for _ in range(nL)])
        self.cr16, self.cr32, self.cr64 = mk_cr(), mk_cr(), mk_cr()
        # DPT encoder skips (U-Net detail injection)
        self.se4 = nn.Conv2d(ngf * 8, dim, 1)
        self.se3 = nn.Conv2d(ngf * 4, dim, 1)
        self.se2 = nn.Conv2d(ngf * 2, dim, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.refine32 = Refine(dim); self.refine64 = Refine(dim)
        self.lsa32 = LocalSphericalAttention(dim, heads, 32, 64, getattr(cfg, "raydpt_win32", 5))
        self.lsa64 = LocalSphericalAttention(dim, heads, 64, 128, getattr(cfg, "raydpt_win64", 3))
        self.coarse_head = nn.Conv2d(dim, 1, 1)
        self.head = nn.Sequential(conv_bn(dim, ngf), conv_bn(ngf, ngf), nn.Conv2d(ngf, 1, 3, 1, 1))
        self.lite = getattr(cfg, "raydpt_lite", False)        # 2-scale (32,64) lite variant
        # full-decode: LEARNED upsample 64x128 -> 256x512 (+e1 skip) instead of bilinear x4.
        # Improves RMSE/d1 (full-res detail) -- the honest-metric lever.
        self.full_decode = getattr(cfg, "raydpt_full_decode", False)
        if self.full_decode:
            self.proj_fd = conv_bn(dim, ngf)                  # dim->ngf at 64x128
            self.se1 = nn.Conv2d(ngf, ngf, 1)                 # e1 (128x256) DPT skip
            self.dec1 = Refine(ngf)                           # at 128x256
            self.dec2 = nn.Sequential(conv_bn(ngf, ngf), Refine(ngf))   # at 256x512
            self.head_fd = nn.Conv2d(ngf, 1, 3, 1, 1)

    def _cross(self, rp, rf, blocks, kv, B, h, w):
        q = rp(rf)[None].expand(B, -1, -1)
        for blk in blocks:
            q = blk(q, kv)
        return q.transpose(1, 2).reshape(B, -1, h, w)

    def forward(self, spec, coarse_feat=None, sh_basis=None):
        B = spec.size(0)
        e1 = self.enc.e1(spec); e2 = self.enc.e2(e1); e3 = self.enc.e3(e2); e4 = self.enc.e4(e3)
        kv4 = self.kv_e4(e4.flatten(2).transpose(1, 2))        # (B,512,dim)
        if self.lite:
            # 2-scale lite: ONE ray cross-attn at 32x64 (Q32 <- e4), e3/e2 projection
            # skips + local spherical attn. Isolates the DPT-fusion / ray-grid gain.
            F32 = self._cross(self.rp32, self.rf32, self.cr32, kv4, B, 32, 64)
            m = F32 + self.se3(e3)                              # 32x64
            d_c = torch.sigmoid(self.coarse_head(F.adaptive_avg_pool2d(m, (16, 32))))
            x = self.lsa32(self.refine32(m))                    # 32x64
            x = self.lsa64(self.refine64(self.up(x) + self.se2(e2)))   # 64x128
        else:
            kv3 = self.kv_e3(e3.flatten(2).transpose(1, 2))     # (B,2048,dim)
            F16 = self._cross(self.rp16, self.rf16, self.cr16, kv4, B, 16, 32)
            F32 = self._cross(self.rp32, self.rf32, self.cr32, kv3, B, 32, 64)
            F64 = self._cross(self.rp64, self.rf64, self.cr64, kv4, B, 64, 128)
            m16 = F16 + self.se4(e4)                             # 16x32
            d_c = torch.sigmoid(self.coarse_head(m16))          # (B,1,16,32) coarse layout
            x = self.lsa32(self.refine32(self.up(m16) + F32 + self.se3(e3)))   # 32x64
            x = self.lsa64(self.refine64(self.up(x) + F64 + self.se2(e2)))     # 64x128
        if self.full_decode:                                   # LEARNED upsample 64x128 -> 256x512
            xf = self.up(self.proj_fd(x))                       # 128x256, ngf
            xf = self.dec1(xf + self.se1(e1))                  # + e1 skip
            xf = self.dec2(self.up(xf))                        # 256x512, ngf
            D = torch.sigmoid(self.head_fd(xf))
        else:
            D = torch.sigmoid(self.head(x))
            D = F.interpolate(D, (self.H, self.W), mode="bilinear", align_corners=False)
        return {"D": D, "D0": D, "extras": {"D_coarse": d_c}}


def build_model(cfg):
    return RayDPT(build_model_cfg(cfg))


# ============================================================
# Loss / objective helpers
# ============================================================

def gaussian_blur_erp(x, sigma):
    """Separable Gaussian low-pass on (B,1,H,W): reflect pad on height, circular
    (azimuth wraps) pad on width."""
    k = int(2 * round(3 * sigma) + 1)
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="circular"), g.view(1, 1, 1, k))
    return x


def masked_mae(D, gt, mask):
    return ((D - gt).abs() * mask).sum() / mask.sum().clamp(min=1e-6)


def silog_loss(D, gt, mask, lam=0.85, eps=1e-6):
    """Scale-invariant log loss (Eigen): sqrt(mean(g^2) - lam*mean(g)^2), g=log D - log gt.
    Penalises log-error variance -> rewards correct relative STRUCTURE; complements the
    relative term (which targets relative magnitude) with a far-pixel-gentle gradient."""
    g = (torch.log(D.clamp(min=eps)) - torch.log(gt.clamp(min=eps))) * mask
    n = mask.sum().clamp(min=1e-6)
    var = (g ** 2).sum() / n - lam * ((g.sum() / n) ** 2)
    return torch.sqrt(var.clamp(min=1e-8))


def composite_loss(out, gt, mask, mcfg):
    """Band-limited objective: dense masked-MAE + coarse-layout + low-pass.
    gt / out['D'] are normalised depth in [0,1]. Returns (loss, parts)."""
    main = masked_mae(out["D"], gt, mask)
    loss = mcfg.w_dense * main
    # relative-error term: directly targets the eval metric ABS_REL = mean(|D-gt|/gt).
    # gt.clamp(0.01)=0.1m floor matches the metric's near-depth regime; max_depth cancels,
    # so in normalised space this term equals ABS_REL. Pushes near-field (small-gt) accuracy
    # that uniform MAE under-weights -> aims to break the ABS_REL<->RMSE LR tradeoff.
    rel = (((out["D"] - gt).abs() / gt.clamp(min=0.01)) * mask).sum() / mask.sum().clamp(min=1e-6)
    loss = loss + mcfg.w_rel * rel
    # scale-invariant log term (structure), stacked on the relative term (magnitude).
    sil = silog_loss(out["D"], gt, mask)
    loss = loss + mcfg.w_silog * sil
    chh, chw = mcfg.coarse_head_h, mcfg.coarse_head_w
    gt_c = F.adaptive_avg_pool2d(gt, (chh, chw))
    m_c = F.adaptive_avg_pool2d(mask, (chh, chw))
    dco = out["extras"].get("D_coarse")
    if dco is not None and dco.shape[-2:] == gt_c.shape[-2:]:
        lc = masked_mae(dco, gt_c, m_c)
    else:
        lc = masked_mae(F.adaptive_avg_pool2d(out["D"], (chh, chw)), gt_c, m_c)
    ll = masked_mae(gaussian_blur_erp(out["D"], 3.0), gaussian_blur_erp(gt, 3.0), mask)
    loss = loss + mcfg.w_coarse_layout * lc + mcfg.w_low * ll
    return loss, {"mae": float(main.detach()), "rel": float(rel.detach()),
                  "sil": float(sil.detach()), "lc": float(lc.detach()),
                  "llow": float(ll.detach())}


# ============================================================
# Visualization
# ============================================================

def save_visualizations(vis_data, epoch, vis_dir, max_depth):
    """Save GT-vs-pred depth panels (+ RGB if available) for a few val samples."""
    for j, v in enumerate(vis_data):
        gt, pred = v['gt_depth'].squeeze(), v['pred_depth'].squeeze()
        rgb = v.get('gt_rgb')
        n = 4 if rgb is not None else 3
        fig, ax = plt.subplots(1, n, figsize=(4 * n, 3))
        col = 0
        if rgb is not None:
            ax[col].imshow(rgb); ax[col].set_title('RGB'); col += 1
        ax[col].imshow(gt, vmin=0, vmax=max_depth, cmap='turbo'); ax[col].set_title('GT'); col += 1
        ax[col].imshow(pred, vmin=0, vmax=max_depth, cmap='turbo'); ax[col].set_title('Pred'); col += 1
        ax[col].imshow(np.abs(gt - pred), vmin=0, vmax=max_depth, cmap='magma'); ax[col].set_title('|err|')
        for a in ax:
            a.axis('off')
        fig.suptitle(v.get('key', ''))
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f'ep{epoch:03d}_{j:02d}.png'), dpi=80)
        plt.close(fig)


# ============================================================
# Training
# ============================================================

def _build_optimizer(model, cfg):
    lr, wd = cfg.mode.learning_rate, cfg.mode.weight_decay
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.mode.optimizer == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif cfg.mode.optimizer == 'Adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)


@torch.no_grad()
def evaluate(model, loader, mcfg, device, max_depth, collect_vis=None,
             dataset=None, dataset_dir=None, depth_type='erp'):
    """Run the model over `loader`; return (mean_errors, val_loss, vis_data).

    mean_errors = mean over samples of compute_errors() -> [abs_rel, rmse, a1, a2, a3, log10, mae].
    """
    model.eval()
    errors, val_losses, vis_data = [], [], []
    seen = 0
    for b in loader:
        spec = b["spec"].to(device, non_blocking=True)
        gt = b["depth"].to(device); mask = b["mask"].to(device)
        out = model(spec)
        loss, _ = composite_loss(out, gt, mask, mcfg)
        val_losses.append(float(loss.detach()))
        pred_m = (out["D"] * max_depth).cpu().numpy()
        gt_m = (gt * max_depth).cpu().numpy()
        for k in range(pred_m.shape[0]):
            errors.append(compute_errors(gt_m[k, 0], pred_m[k, 0]))
            if collect_vis is not None and seen in collect_vis:
                key = b["key"][k]
                gt_rgb = None
                if dataset_dir is not None:
                    scene, sidx = key.split('/')
                    gt_rgb = load_gt_rgb(dataset_dir, scene, sidx, depth_type,
                                         gt_m.shape[2], gt_m.shape[3])
                vis_data.append({'gt_depth': gt_m[k, 0], 'pred_depth': pred_m[k, 0],
                                 'gt_rgb': gt_rgb, 'key': key})
            seen += 1
    mean_errors = np.array(errors).mean(0) if errors else np.zeros(7)
    return mean_errors, float(np.mean(val_losses)) if val_losses else 0.0, vis_data


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{torch.cuda.device_count()} {device} device(s)")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    mcfg = build_model_cfg(cfg)
    max_depth = float(cfg.dataset.max_depth)

    # Data
    train_set, train_loader = make_dataloader(cfg, 'train')
    val_set, val_loader = make_dataloader(cfg, 'val', shuffle=False)
    print(f'Train: {len(train_set)} samples, Val: {len(val_set)} samples')

    # Model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {cfg.model.name} ({total_params:.2f}M params)')

    # Optimizer + warmup-cosine schedule
    optimizer = _build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.mode.epochs * steps_per_epoch
    warm = steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))

    # Output directories
    project_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = cfg.mode.experiment_name
    ckpt_dir = os.path.join(project_dir, 'checkpoints', experiment_name)
    vis_dir = os.path.join(project_dir, 'outputs', experiment_name, 'visualizations')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f'Experiment: {experiment_name}')
    print(f'Checkpoints: {ckpt_dir}')

    # Resume
    start_epoch = 1
    if cfg.mode.checkpoints is not None:
        ckpt = torch.load(os.path.join(ckpt_dir, f'checkpoint_{cfg.mode.checkpoints}.pth'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    # Which val samples to visualize
    n_vis = min(12, len(val_set))
    vis_indices = set(np.linspace(0, max(0, len(val_set) - 1), n_vis, dtype=int).tolist())

    best_abs_rel = float('inf')
    best_score = float('inf')          # composite (abs_rel + rmse, normalised) for model selection
    best_rmse = float('inf')
    dataset_dir = cfg.dataset.dataset_dir
    depth_type = cfg.dataset.depth_type

    training_start = time.time()
    for epoch in range(start_epoch, cfg.mode.epochs + 1):
        model.train()
        t0 = time.time()
        accum = {}
        n_batches = len(train_loader)
        for i, b in enumerate(train_loader):
            spec = b["spec"].to(device, non_blocking=True)
            gt = b["depth"].to(device); mask = b["mask"].to(device)
            if cfg.mode.flip_aug:                    # L/R mirror aug (per-sample, p=0.5)
                fm = torch.rand(spec.size(0), device=device) < 0.5
                if fm.any():
                    spec = spec.clone(); gt = gt.clone(); mask = mask.clone()
                    spec[fm] = swap_audio_lr(spec[fm])
                    gt[fm] = torch.flip(gt[fm], dims=[-1])
                    mask[fm] = torch.flip(mask[fm], dims=[-1])
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(spec)
                loss, parts = composite_loss(out, gt, mask, mcfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            accum['total'] = accum.get('total', 0.0) + float(loss.detach())
            for k, v in parts.items():
                accum[k] = accum.get(k, 0.0) + v

            if (i + 1) % max(1, n_batches // 5) == 0 or (i + 1) == n_batches:
                prog = (i + 1) / n_batches * 100
                print(f'  Epoch {epoch} [{i+1}/{n_batches} {prog:.0f}%] '
                      f'Loss: {accum["total"]/(i+1):.4f} '
                      f'mae:{accum["mae"]/(i+1):.4f} rel:{accum["rel"]/(i+1):.4f} '
                      f'lc:{accum["lc"]/(i+1):.4f} '
                      f'low:{accum["llow"]/(i+1):.4f}', flush=True)

        epoch_loss = accum['total'] / n_batches
        print(f'Epoch [{epoch}/{cfg.mode.epochs}] Loss: {epoch_loss:.4f} '
              f'Time: {time.time()-t0:.1f}s LR: {scheduler.get_last_lr()[0]:.6f}')

        # --- Validation ---
        if epoch % cfg.mode.validation_iter == 0:
            mean_errors, val_loss, vis_data = evaluate(
                model, val_loader, mcfg, device, max_depth,
                collect_vis=vis_indices, dataset=val_set,
                dataset_dir=dataset_dir, depth_type=depth_type)
            abs_rel = mean_errors[0]
            print(f'  Val Loss: {val_loss:.4f} | '
                  f'ABS_REL: {abs_rel:.4f} RMSE: {mean_errors[1]:.4f} '
                  f'd1: {mean_errors[2]:.4f} d2: {mean_errors[3]:.4f} d3: {mean_errors[4]:.4f}')

            if vis_data:
                save_visualizations(vis_data, epoch, vis_dir, max_depth)
                print(f'  Saved {len(vis_data)} visualizations')

            rmse = mean_errors[1]
            # COMPOSITE selection: balance ABS_REL and RMSE (normalised by typical scales,
            # equal weight). ABS_REL alone systematically picks the epoch where RMSE spikes
            # (the two anti-correlate epoch-to-epoch), so it saved RMSE-broken checkpoints.
            score = abs_rel / 0.4 + rmse / 1.6
            if score < best_score:
                best_score = score
                best_abs_rel = abs_rel
                best_rmse = rmse
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_abs_rel': best_abs_rel, 'best_rmse': best_rmse,
                            'best_score': best_score, 'cfg_model': vars(mcfg)},
                           os.path.join(ckpt_dir, 'best_model.pth'))
                print(f'  >> Best model saved (score {best_score:.4f} | '
                      f'ABS_REL {best_abs_rel:.4f} RMSE {best_rmse:.4f})')

        # Time budget check
        elapsed = time.time() - training_start
        if elapsed >= TIME_BUDGET:
            print(f'\nTime budget reached ({elapsed:.1f}s >= {TIME_BUDGET}s). Stopping.')
            break

    total_time = time.time() - training_start
    print(f'\nTraining complete. Best (composite) ABS_REL: {best_abs_rel:.4f} '
          f'RMSE: {best_rmse:.4f} score: {best_score:.4f}')
    print(f'training_seconds: {total_time:.1f}')
    if device.type == 'cuda':
        print(f'peak_vram_mb: {torch.cuda.max_memory_allocated() / 1e6:.1f}')


# ============================================================
# Testing
# ============================================================

def test(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{torch.cuda.device_count()} {device} device(s)")

    mcfg = build_model_cfg(cfg)
    max_depth = float(cfg.dataset.max_depth)
    eval_on = cfg.mode.eval_on

    eval_set, eval_loader = make_dataloader(cfg, eval_on, shuffle=False)
    print(f'Eval [{eval_on}]: {len(eval_set)} samples')

    model = build_model(cfg).to(device)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = cfg.mode.experiment_name
    ckpt_path = os.path.join(project_dir, 'checkpoints', experiment_name, 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f'Loaded {ckpt_path} (epoch {ckpt.get("epoch", "?")}, '
          f'best ABS_REL {ckpt.get("best_abs_rel", float("nan")):.4f})')

    vis_dir = os.path.join(project_dir, 'outputs', experiment_name, 'test_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    vis_every = cfg.mode.vis_every
    vis_indices = set(range(0, len(eval_set), vis_every)) if vis_every else set()

    mean_errors, _, vis_data = evaluate(
        model, eval_loader, mcfg, device, max_depth,
        collect_vis=vis_indices, dataset=eval_set,
        dataset_dir=cfg.dataset.dataset_dir, depth_type=cfg.dataset.depth_type)
    print(f'[{eval_on}] ABS_REL: {mean_errors[0]:.4f} RMSE: {mean_errors[1]:.4f} '
          f'd1: {mean_errors[2]:.4f} d2: {mean_errors[3]:.4f} d3: {mean_errors[4]:.4f} '
          f'log10: {mean_errors[5]:.4f} MAE: {mean_errors[6]:.4f}')
    if vis_data:
        save_visualizations(vis_data, 0, vis_dir, max_depth)
        print(f'Saved {len(vis_data)} visualizations -> {vis_dir}')


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='RayDPT: Depth from Binaural Echoes')

    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--eval-on', type=str, default='test', choices=['test', 'val'])

    p.add_argument('--dataset-dir', type=str,
                   default='/home/rvi-lab/workspace/sound-spaces/dataset_simplified',
                   help='Path to SoundSpaces dataset')

    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'])
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--in-ch', type=int, default=5, choices=[2, 3, 5],
                   help='5=RIR spatial feature [logL,logR,ILD,cosIPD,sinIPD] (default); '
                        '2=log-mag binaural; 3=[logL,logR,ILD]')
    p.add_argument('--flip-aug', type=lambda s: s == 'True', default=True,
                   help='L/R mirror augmentation (depth width-flip + channel-aware audio swap)')
    p.add_argument('--raydpt-full-decode', type=lambda s: s == 'True', default=False,
                   help='learned upsample 64x128->256x512 (+e1 skip) instead of bilinear x4')
    p.add_argument('--raydpt-lite', type=lambda s: s == 'True', default=False,
                   help='2-scale (32,64) lite RayDPT variant')

    p.add_argument('--experiment-name', type=str, default='raydpt_5chflip')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Checkpoint epoch to resume')
    p.add_argument('--vis-every', type=int, default=100,
                   help='Visualize every N samples during test (0=skip)')

    return p.parse_args()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')

    args = parse_args()
    cfg = make_config(args)

    print('=' * 60)
    print(f'RayDPT — mode={args.mode}')
    print(f'Dataset: {args.dataset_dir}')
    print(f'Batch size: {args.batch_size}, LR: {args.lr}, Optimizer: {args.optimizer}')
    print('=' * 60)

    if args.mode == 'train':
        train(cfg)
    elif args.mode == 'test':
        test(cfg)
