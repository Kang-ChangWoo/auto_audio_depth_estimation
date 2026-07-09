#!/usr/bin/env python3
"""
BASELINE RUNNER — trains the BatVision U-Net baseline (`base/unet_baseline.py`) under
the EXACT same harness as `train.py`.

This file is a structural clone of `train.py`: identical config, composite
"coarse-arch" objective (dense masked-MAE + coarse-layout + low-pass), training /
testing loops, CLI, and fixed `prepare.py` data + evaluation harness. The ONLY
difference is the model: `build_model` returns the plain pix2pix / CycleGAN
encoder->decoder U-Net (`unet_256`, 8 downs) from AmandineBtto/Batvision-Dataset,
instead of RayDPT.

It exists to establish the baseline number that "my model" (`train.py`) must beat,
measured on the same split / target / metric / selection composite.

Model (256x512):
    spec (in_ch) -> UnetGenerator (unet_256, 8 downs, ngf=64, BatchNorm) -> sigmoid ERP depth.

Usage:
    python run_base.py --mode train [options]
    python run_base.py --mode test  [options]
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
    make_dataloader, compute_errors, load_gt_rgb, swap_audio_lr, build_channel_names,
)
from base.batvision import build_batvision_model


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
    in_ch = len(build_channel_names(
        args.use_log, args.feat_L, args.feat_R, args.feat_ILD,
        args.feat_cosIPD, args.feat_sinIPD))
    cfg = Cfg(
        dataset=Cfg(
            name='soundspaces',
            dataset_dir=args.dataset_dir,
            split_ratio=[0.8, 0.1, 0.1],
            split_seed=42,
            depth_type='erp',
            images_size=[256, 512],
            max_depth=10.0,
            in_ch=in_ch,                # derived from the cue toggles below
            sample_rate=48000,
            # --- input representation: named cue toggles + log switch ---
            use_log=args.use_log,
            feat_L=args.feat_L,
            feat_R=args.feat_R,
            feat_ILD=args.feat_ILD,
            feat_cosIPD=args.feat_cosIPD,
            feat_sinIPD=args.feat_sinIPD,
            audio_window_m=10.0,
        ),
        model=Cfg(
            name='batvision',
            generator='unet_256',
            ngf=64,
            # composite-loss head / weights
            coarse_head_h=16,
            coarse_head_w=32,
            w_dense=1.0,
            w_coarse_layout=1.0,
            w_low=0.5,
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
            max_iters=args.max_iters,             # >0: stop each epoch after N train iters (smoke/debug)
            max_val_batches=args.max_val_batches,  # >0: evaluate only N val batches (smoke/debug)
        ),
    )
    return cfg


def build_model_cfg(cfg):
    """Flatten the nested cfg into the attribute namespace the loss config expects
    (coarse_head_h/w, w_dense, w_coarse_layout, w_low, img_h/w, in_ch, max_depth)."""
    h, w = int(cfg.dataset.images_size[0]), int(cfg.dataset.images_size[1])
    m = vars(cfg.model).copy()
    m.update(img_h=h, img_w=w,
             in_ch=int(getattr(cfg.dataset, 'in_ch', 2)),
             max_depth=float(cfg.dataset.max_depth))
    return SimpleNamespace(**m)


def build_model(cfg):
    # BatVision plain U-Net (unet_256). Takes the nested cfg directly
    # (reads cfg.dataset.images_size / in_ch and cfg.model.ngf / generator).
    return build_batvision_model(cfg)


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


def composite_loss(out, gt, mask, mcfg):
    """Band-limited objective: dense masked-MAE + coarse-layout + low-pass.
    gt / out['D'] are normalised depth in [0,1]. Returns (loss, parts)."""
    main = masked_mae(out["D"], gt, mask)
    loss = mcfg.w_dense * main
    chh, chw = mcfg.coarse_head_h, mcfg.coarse_head_w
    # METRIC/LOSS FIX (2026-July validation reset): the coarse-layout and low-pass loss TARGETS are now
    # computed with MASK-WEIGHTED pooling. The previous code averaged/blurred gt over ALL pixels including
    # INVALID ones (gt=0 where mask=0), which diluted the coarse/low target toward 0 in any cell/region
    # containing invalid pixels -> the aux losses were trained against a corrupted target. The correct
    # target is the mean of VALID depths only: sum(gt*mask)/sum(mask).
    m_c = F.adaptive_avg_pool2d(mask, (chh, chw))
    gt_c = F.adaptive_avg_pool2d(gt * mask, (chh, chw)) / m_c.clamp(min=1e-6)   # mask-weighted coarse target
    dco = out["extras"].get("D_coarse")
    if dco is not None and dco.shape[-2:] == gt_c.shape[-2:]:
        lc = masked_mae(dco, gt_c, m_c)
    else:
        lc = masked_mae(F.adaptive_avg_pool2d(out["D"], (chh, chw)), gt_c, m_c)
    gt_low = gaussian_blur_erp(gt * mask, 3.0) / gaussian_blur_erp(mask, 3.0).clamp(min=1e-6)  # mask-weighted low target
    ll = masked_mae(gaussian_blur_erp(out["D"], 3.0), gt_low, mask)
    loss = loss + mcfg.w_coarse_layout * lc + mcfg.w_low * ll
    return loss, {"mae": float(main.detach()), "lc": float(lc.detach()),
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
             dataset=None, dataset_dir=None, depth_type='erp', max_batches=0):
    """Run the model over `loader`; return (mean_errors, val_loss, vis_data).

    mean_errors = mean over samples of compute_errors() -> [abs_rel, rmse, a1, a2, a3, log10, mae].
    max_batches>0 evaluates only that many batches (smoke/debug).
    """
    model.eval()
    errors, val_losses, vis_data = [], [], []
    seen = 0
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
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
    best_score = float('inf')   # honest composite (model selection)
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
                    spec[fm] = swap_audio_lr(spec[fm], train_set.channel_names)
                    gt[fm] = torch.flip(gt[fm], dims=[-1])
                    mask[fm] = torch.flip(mask[fm], dims=[-1])
            out = model(spec)
            loss, parts = composite_loss(out, gt, mask, mcfg)

            optimizer.zero_grad()
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
                      f'mae:{accum["mae"]/(i+1):.4f} lc:{accum["lc"]/(i+1):.4f} '
                      f'low:{accum["llow"]/(i+1):.4f}', flush=True)

            if cfg.mode.max_iters and (i + 1) >= cfg.mode.max_iters:
                print(f'  [smoke] stopping epoch after {i+1} iters (--max-iters)', flush=True)
                break

        n_ran = min(n_batches, cfg.mode.max_iters) if cfg.mode.max_iters else n_batches
        epoch_loss = accum['total'] / max(1, n_ran)
        print(f'Epoch [{epoch}/{cfg.mode.epochs}] Loss: {epoch_loss:.4f} '
              f'Time: {time.time()-t0:.1f}s LR: {scheduler.get_last_lr()[0]:.6f}')

        # --- Validation ---
        if epoch % cfg.mode.validation_iter == 0:
            mean_errors, val_loss, vis_data = evaluate(
                model, val_loader, mcfg, device, max_depth,
                collect_vis=vis_indices, dataset=val_set,
                dataset_dir=dataset_dir, depth_type=depth_type,
                max_batches=cfg.mode.max_val_batches)
            abs_rel = mean_errors[0]; rmse = mean_errors[1]; d1 = mean_errors[2]
            print(f'  Val Loss: {val_loss:.4f} | '
                  f'ABS_REL: {abs_rel:.4f} RMSE: {rmse:.4f} '
                  f'd1: {d1:.4f} d2: {mean_errors[3]:.4f} d3: {mean_errors[4]:.4f}')

            if vis_data:
                save_visualizations(vis_data, epoch, vis_dir, max_depth)
                print(f'  Saved {len(vis_data)} visualizations')

            # HONEST-WEIGHTED composite for model selection. RMSE + d1 dominate (not directly optimised
            # -> trustworthy); ABS_REL is directly optimisable (gameable) AND varies most across runs, so
            # it is DE-WEIGHTED (2026-July: effective per-unit coeff 0.75 -> 0.35). Lower is better.
            score = rmse / 1.6 + (1.0 - d1) / 0.46 + 0.35 * abs_rel
            if score < best_score:
                best_score = score; best_abs_rel = abs_rel
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_score': best_score, 'best_abs_rel': best_abs_rel,
                            'cfg_model': vars(mcfg)},
                           os.path.join(ckpt_dir, 'best_model.pth'))
                print(f'  >> Best model saved (score {best_score:.4f} | '
                      f'ABS_REL {abs_rel:.4f} RMSE {rmse:.4f} d1 {d1:.4f})')

        # Time budget check
        elapsed = time.time() - training_start
        if elapsed >= TIME_BUDGET:
            print(f'\nTime budget reached ({elapsed:.1f}s >= {TIME_BUDGET}s). Stopping.')
            break

    total_time = time.time() - training_start
    print(f'\nTraining complete. Best (composite) score: {best_score:.4f} ABS_REL: {best_abs_rel:.4f}')
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
    p = argparse.ArgumentParser(description='BatVision U-Net baseline: Depth from Binaural Echoes')

    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--eval-on', type=str, default='test', choices=['test', 'val'])

    p.add_argument('--dataset-dir', type=str,
                   default='/home/rvi-lab/workspace/sound-spaces/dataset_simplified',
                   help='Path to SoundSpaces dataset')

    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'])
    p.add_argument('--num-workers', type=int, default=16)

    # --- input representation: named binaural cues (each on/off) + log switch ---
    _bool = lambda s: s == 'True'
    p.add_argument('--use-log', type=_bool, default=True,
                   help='log1p-compress the L/R magnitude channels (logL/logR vs raw L/R)')
    p.add_argument('--feat-L', type=_bool, default=True, help='include left magnitude channel')
    p.add_argument('--feat-R', type=_bool, default=True, help='include right magnitude channel')
    p.add_argument('--feat-ILD', type=_bool, default=True, help='include ILD = log|L|-log|R|')
    p.add_argument('--feat-cosIPD', type=_bool, default=True, help='include cos(IPD)')
    p.add_argument('--feat-sinIPD', type=_bool, default=True, help='include sin(IPD)')
    p.add_argument('--flip-aug', type=_bool, default=True,
                   help='L/R mirror augmentation (depth width-flip + channel-aware audio swap)')

    p.add_argument('--experiment-name', type=str, default='batvision')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Checkpoint epoch to resume')
    p.add_argument('--vis-every', type=int, default=100,
                   help='Visualize every N samples during test (0=skip)')
    p.add_argument('--max-iters', type=int, default=0,
                   help='Smoke/debug: stop each epoch after N training iterations (0=full epoch)')
    p.add_argument('--max-val-batches', type=int, default=0,
                   help='Smoke/debug: evaluate only N validation batches (0=full val set)')

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
    print(f'BatVision U-Net — mode={args.mode}')
    print(f'Dataset: {args.dataset_dir}')
    print(f'Batch size: {args.batch_size}, LR: {args.lr}, Optimizer: {args.optimizer}')
    print('=' * 60)

    if args.mode == 'train':
        train(cfg)
    elif args.mode == 'test':
        test(cfg)
