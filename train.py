#!/usr/bin/env python3
"""
AudioDepthFOA V2: Joint depth estimation + SH Order-5 prediction from binaural echoes.

Consolidated training, validation, and testing script.

Usage:
    python train.py --mode train [options]
    python train.py --mode test  [options]
"""

import argparse
import math
import functools
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import lpmv
from scipy.special import factorial as sp_factorial
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare import (
    SoundSpacesDataset, make_dataloader, compute_errors, compute_foa_errors,
    load_gt_rgb, sh_basis_matrix, reconstruct_energy_maps,
    _acn_to_nm, _sn3d_norm, _real_sh_sn3d_np,
)


# ============================================================
# Constants (fixed, do not modify)
# ============================================================

TIME_BUDGET = 3600  # training time budget in seconds (1 hour)

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
            input_type='echoes',
            audio_format='spectrogram',
            depth_type='erp',
            preprocess='resize',
            depth_norm=True,
            images_size=[256, 512],
            min_depth=0.01,
            max_depth=10.0,
            use_ambisonic=True,
        ),
        model=Cfg(
            name='audio_depth_foa_v2',
            generator='unet_256',
            proj_dim=128,
            foa_dim=4,
            sh_order=5,
            scale_shift_hidden=128,
            scale_shift_layers=2,
            depth_weight=1.0,
            foa_weight=0.2,
            foa_use_cosine=True,
            foa_cosine_weight=0.2,
            hist_weight=0.2,
            latent_reg_weight=0.001,
            foa_freeze_epochs=args.foa_freeze_epochs,
        ),
        mode=Cfg(
            mode=args.mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            validation=True,
            validation_iter=2,
            saving_checkpoints=10,
            shuffle=True,
            num_threads=args.num_workers,
            checkpoints=args.checkpoint,
            use_l1=True,
            use_berhu=True,
            use_silog=True,
            use_gradient=True,
            use_ssim=False,
            w_l1=1.0,
            w_berhu=1.0,
            w_gradient=0.5,
            w_silog=0.5,
            w_ssim=0.0,
            experiment_name=args.experiment_name,
            eval_on=args.eval_on,
            vis_every=args.vis_every,
        ),
    )
    return cfg


# ============================================================
# SH basis computation (ACN / SN3D)
# ============================================================

def sh_basis_erp(max_order, H, W, dtype=torch.float32):
    """Compute real SH basis functions up to given order on ERP grid.
    Returns tensor of shape [(max_order+1)^2, H, W]."""
    n_ch = (max_order + 1) ** 2
    theta = np.linspace(0, np.pi, H)
    phi = np.linspace(-np.pi, np.pi, W)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    elevation = np.pi / 2 - theta_grid
    azimuth = phi_grid

    basis = np.zeros((n_ch, H, W), dtype=np.float64)
    for q in range(n_ch):
        basis[q] = _real_sh_sn3d_np(q, elevation, azimuth)
    return torch.from_numpy(basis).to(dtype)


# ============================================================
# Model: DeepScaleShift
# ============================================================

class DeepScaleShift(nn.Module):
    """MLP-based per-channel affine transform with residual + gating."""

    def __init__(self, n_channels=36, hidden_dim=256, n_hidden_layers=4, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        layers = [nn.LayerNorm(n_channels)]
        in_dim = n_channels
        for i in range(n_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            if dropout > 0 and i < n_hidden_layers - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_channels))
        self.mlp = nn.Sequential(*layers)

        self.gamma = nn.Parameter(torch.ones(n_channels))
        self.beta = nn.Parameter(torch.zeros(n_channels))
        self.gate = nn.Parameter(torch.zeros(n_channels))
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)
        mlp_out = self.mlp(x)
        alpha = torch.sigmoid(self.gate).unsqueeze(0)
        return (1 - alpha) * residual + alpha * mlp_out


# ============================================================
# Model: AudioDepthFOAV2Generator (UNet + SH branch)
# ============================================================

class AudioDepthFOAV2Generator(nn.Module):
    """UNet encoder-decoder with SH5 auxiliary branch."""

    def __init__(self, cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64,
                 use_dropout=False, proj_dim=128, foa_dim=4, sh_order=5,
                 scale_shift_hidden=256, scale_shift_layers=4,
                 H_erp=256, W_erp=512):
        super().__init__()
        self.num_downs = num_downs
        self.depth_norm = cfg.dataset.depth_norm
        self.proj_dim = proj_dim
        self.foa_dim = foa_dim
        self.sh_order = sh_order
        self.sh_dim = (sh_order + 1) ** 2

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        use_bias = False

        basis = sh_basis_erp(sh_order, H_erp, W_erp, dtype=torch.float32)
        self.register_buffer("sh_basis", basis, persistent=False)

        self.scale_shift = DeepScaleShift(
            n_channels=self.sh_dim,
            hidden_dim=scale_shift_hidden,
            n_hidden_layers=scale_shift_layers,
        )

        # Encoder
        self.enc0 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        encoder_layers = []
        in_ch = ngf
        for i in range(1, num_downs - 1):
            out_ch = min(ngf * (2 ** i), ngf * 8)
            encoder_layers.append(nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias),
                norm_layer(out_ch),
            ))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoder_layers)
        self.enc_inner = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, ngf * 8, 4, 2, 1),
        )

        # SH Branch
        feat_dim = ngf * 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.audio_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
        self.foa_head = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, foa_dim),
        )
        hoa_dim = self.sh_dim - foa_dim
        self.hoa_head = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, hoa_dim),
        )

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 8),
        ))
        for _ in range(num_downs - 5):
            layers = [
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=use_bias),
                norm_layer(ngf * 8),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            decoder_layers.append(nn.Sequential(*layers))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
        ))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
        ))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
        ))
        self.decoders = nn.ModuleList(decoder_layers)

        if self.depth_norm:
            self.dec_outer = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.dec_outer = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
                nn.ReLU(),
            )

    def reconstruct_from_coeffs(self, coeffs):
        return (coeffs[:, :, None, None] * self.sh_basis[None]).sum(dim=1, keepdim=True)

    def project_depth_to_sh(self, depth, eps=1e-6):
        basis = self.sh_basis
        d = depth[:, 0:1, :, :]
        H = basis.shape[1]
        theta = torch.linspace(0, math.pi, H, device=basis.device, dtype=basis.dtype)
        sin_weight = torch.sin(theta)[:, None]
        w = sin_weight[None, None, :, :]
        num = (w * d * basis[None]).sum(dim=(2, 3))
        den = (w * basis[None] ** 2).sum(dim=(2, 3)) + eps
        return num / den

    def forward(self, x, return_hist_maps=False, ambi_maps=None):
        if ambi_maps is not None:
            x = torch.cat([x, ambi_maps], dim=1)
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)
        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)
        bottleneck = self.enc_inner(h)

        # SH branch
        pooled = self.pool(bottleneck)
        foa_latent = self.audio_proj(pooled)
        pred_foa = self.foa_head(foa_latent)
        pred_hoa = self.hoa_head(foa_latent)
        pred_sh = torch.cat([pred_foa, pred_hoa], dim=1)

        # Decode
        enc_reversed = enc_features[::-1]
        h = self.decoders[0](bottleneck)
        for i in range(len(self.decoders) - 1):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)
        h = torch.cat([enc_reversed[-1], h], dim=1)
        pred_depth = self.dec_outer(h)

        out = {
            "pred_depth": pred_depth,
            "foa_latent": foa_latent,
            "pred_foa": pred_foa,
            "pred_hoa": pred_hoa,
            "pred_sh": pred_sh,
        }

        if return_hist_maps:
            sh_aligned = self.scale_shift(pred_sh)
            energy_recon_aligned = self.reconstruct_from_coeffs(sh_aligned)
            depth_sh_coeffs = self.project_depth_to_sh(pred_depth)
            depth_sh = self.reconstruct_from_coeffs(depth_sh_coeffs)
            out["energy_recon_aligned"] = energy_recon_aligned
            out["depth_sh"] = depth_sh
            out["depth_sh_coeffs"] = depth_sh_coeffs
            out["sh_aligned"] = sh_aligned

        return out




# ============================================================
# Loss functions
# ============================================================

class SILogLoss(nn.Module):
    def __init__(self, variance_weight=0.5):
        super().__init__()
        self.variance_weight = variance_weight

    def forward(self, pred, gt):
        mask = gt > 0
        pred, gt = pred[mask], gt[mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        log_diff = torch.log(pred + 1e-6) - torch.log(gt + 1e-6)
        return torch.mean(log_diff ** 2) - self.variance_weight * (torch.mean(log_diff) ** 2)


class GradientLoss(nn.Module):
    def forward(self, pred, gt):
        mask = (gt > 0).float()
        pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        gt_dx = gt[:, :, :, :-1] - gt[:, :, :, 1:]
        gt_dy = gt[:, :, :-1, :] - gt[:, :, 1:, :]
        mask_dx = mask[:, :, :, :-1] * mask[:, :, :, 1:]
        mask_dy = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        loss_dx = torch.abs(pred_dx - gt_dx) * mask_dx
        loss_dy = torch.abs(pred_dy - gt_dy) * mask_dy
        return loss_dx.sum() / mask_dx.sum().clamp(min=1) + loss_dy.sum() / mask_dy.sum().clamp(min=1)


class BerHuLoss(nn.Module):
    def forward(self, pred, gt):
        mask = gt > 0
        pred, gt = pred[mask], gt[mask]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        diff = torch.abs(pred - gt)
        c = 0.2 * diff.max().detach()
        l1 = diff[diff <= c]
        l2 = diff[diff > c]
        return (l1.sum() + ((l2 ** 2 + c ** 2) / (2 * c)).sum()) / pred.numel()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, gt):
        mask = (gt > 0).float()
        pred, gt = pred * mask, gt * mask
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        mu_pred = F.avg_pool2d(pred, self.window_size, 1, pad)
        mu_gt = F.avg_pool2d(gt, self.window_size, 1, pad)
        sigma_pred = F.avg_pool2d(pred ** 2, self.window_size, 1, pad) - mu_pred ** 2
        sigma_gt = F.avg_pool2d(gt ** 2, self.window_size, 1, pad) - mu_gt ** 2
        sigma_pg = F.avg_pool2d(pred * gt, self.window_size, 1, pad) - mu_pred * mu_gt
        ssim = ((2 * mu_pred * mu_gt + C1) * (2 * sigma_pg + C2)) / \
               ((mu_pred ** 2 + mu_gt ** 2 + C1) * (sigma_pred + sigma_gt + C2))
        mask_pooled = F.avg_pool2d(mask, self.window_size, 1, pad)
        valid = mask_pooled > 0.5
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return (1 - ssim[valid]).mean()


class DepthLoss(nn.Module):
    def __init__(self, use_l1=True, use_silog=False, use_gradient=False,
                 use_berhu=False, use_ssim=False,
                 w_l1=1.0, w_silog=0.5, w_gradient=0.5, w_berhu=1.0, w_ssim=0.5):
        super().__init__()
        self.use_l1 = use_l1
        self.use_silog = use_silog
        self.use_gradient = use_gradient
        self.use_berhu = use_berhu
        self.use_ssim = use_ssim
        if use_l1:       self.l1 = nn.L1Loss();        self.w_l1 = w_l1
        if use_silog:    self.silog = SILogLoss();      self.w_silog = w_silog
        if use_gradient: self.gradient = GradientLoss(); self.w_gradient = w_gradient
        if use_berhu:    self.berhu = BerHuLoss();      self.w_berhu = w_berhu
        if use_ssim:     self.ssim_loss = SSIMLoss();    self.w_ssim = w_ssim

    def forward(self, pred, gt):
        mask = gt > 0
        loss = torch.tensor(0.0, device=pred.device)
        if self.use_l1:       loss = loss + self.w_l1 * self.l1(pred[mask], gt[mask])
        if self.use_silog:    loss = loss + self.w_silog * self.silog(pred, gt)
        if self.use_gradient: loss = loss + self.w_gradient * self.gradient(pred, gt)
        if self.use_berhu:    loss = loss + self.w_berhu * self.berhu(pred, gt)
        if self.use_ssim:     loss = loss + self.w_ssim * self.ssim_loss(pred, gt)
        return loss


class FOAGuidedLoss(nn.Module):
    def __init__(self, use_cosine=True, cosine_weight=0.1):
        super().__init__()
        self.use_cosine = use_cosine
        self.cosine_weight = cosine_weight

    def forward(self, pred_foa, gt_foa):
        l1 = F.l1_loss(pred_foa, gt_foa)
        if self.use_cosine:
            cos = 1.0 - F.cosine_similarity(pred_foa, gt_foa, dim=-1).mean()
            return l1 + self.cosine_weight * cos
        return l1


class SH5HistogramAlignmentLoss(nn.Module):
    def __init__(self, map_cosine_weight=0.5, coeff_cosine_weight=0.2):
        super().__init__()
        self.map_cosine_weight = map_cosine_weight
        self.coeff_cosine_weight = coeff_cosine_weight

    def _normalize_map(self, x, eps=1e-6):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        x_min = x_flat.min(dim=1, keepdim=True).values
        x_max = x_flat.max(dim=1, keepdim=True).values
        return ((x_flat - x_min) / (x_max - x_min + eps)).view_as(x)

    def forward(self, energy_recon_aligned, depth_sh,
                sh_aligned=None, depth_sh_coeffs=None):
        e_norm = self._normalize_map(energy_recon_aligned)
        d_norm = self._normalize_map(depth_sh)
        loss = F.l1_loss(e_norm, d_norm)

        B = e_norm.shape[0]
        e_flat = e_norm.view(B, -1)
        d_flat = d_norm.view(B, -1)
        loss = loss + self.map_cosine_weight * (1.0 - F.cosine_similarity(e_flat, d_flat, dim=-1).mean())

        if sh_aligned is not None and depth_sh_coeffs is not None:
            loss = loss + self.coeff_cosine_weight * (
                1.0 - F.cosine_similarity(sh_aligned, depth_sh_coeffs, dim=-1).mean())
        return loss


class AudioDepthFOAV2Loss(nn.Module):
    def __init__(self, depth_criterion, foa_criterion=None,
                 depth_weight=1.0, foa_weight=0.1,
                 hist_criterion=None, hist_weight=0.1, latent_reg_weight=0.0):
        super().__init__()
        self.depth_criterion = depth_criterion
        self.foa_criterion = foa_criterion if foa_criterion is not None else FOAGuidedLoss()
        self.hist_criterion = hist_criterion
        self.depth_weight = depth_weight
        self.foa_weight = foa_weight
        self.hist_weight = hist_weight
        self.latent_reg_weight = latent_reg_weight

    def forward(self, outputs, gt_depth, gt_foa,
                gt_depth_sh=None, gt_depth_sh_coeffs=None):
        pred_depth = outputs["pred_depth"]
        pred_foa = outputs["pred_foa"]
        foa_latent = outputs["foa_latent"]

        depth_loss = self.depth_criterion(pred_depth, gt_depth)
        foa_loss = self.foa_criterion(pred_foa, gt_foa)

        ambi_guide_loss = torch.tensor(0.0, device=pred_depth.device)
        depth_follow_loss = torch.tensor(0.0, device=pred_depth.device)
        hist_loss = torch.tensor(0.0, device=pred_depth.device)

        if (self.hist_criterion is not None and self.hist_weight > 0
                and "energy_recon_aligned" in outputs):
            energy_recon = outputs["energy_recon_aligned"]
            depth_sh = outputs["depth_sh"]
            sh_aligned = outputs.get("sh_aligned")
            depth_sh_coeffs = outputs.get("depth_sh_coeffs")

            if gt_depth_sh is not None:
                ambi_guide_loss = self.hist_criterion(
                    energy_recon, gt_depth_sh, sh_aligned, gt_depth_sh_coeffs)
                depth_follow_loss = self.hist_criterion(
                    energy_recon.detach(), depth_sh,
                    sh_aligned.detach() if sh_aligned is not None else None,
                    depth_sh_coeffs)
                hist_loss = ambi_guide_loss + depth_follow_loss
            else:
                hist_loss = self.hist_criterion(energy_recon, depth_sh, sh_aligned, depth_sh_coeffs)

        latent_reg = torch.tensor(0.0, device=pred_depth.device)
        if self.latent_reg_weight > 0:
            latent_reg = (foa_latent ** 2).mean()

        total = (self.depth_weight * depth_loss
                 + self.foa_weight * foa_loss
                 + self.hist_weight * hist_loss
                 + self.latent_reg_weight * latent_reg)

        return {
            "total": total, "depth": depth_loss, "foa": foa_loss,
            "hist_align": hist_loss, "ambi_guide": ambi_guide_loss,
            "depth_follow": depth_follow_loss, "latent_reg": latent_reg,
        }


# ============================================================
# Metrics & Utilities
# ============================================================



def extract_foa_target_from_energy_map(energy_map):
    """Extract (B, 4) FOA target from (B, 4, H, W) covariance-based energy maps.

    Channels: [full, early, late, early-late diff] directional energy.
    """
    return energy_map.mean(dim=(2, 3))


def get_base_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def compute_gt_depth_sh(model, gt_depth):
    base = get_base_model(model)
    with torch.no_grad():
        coeffs = base.project_depth_to_sh(gt_depth)
        sh_map = base.reconstruct_from_coeffs(coeffs)
    return sh_map, coeffs


# ============================================================
# Visualization
# ============================================================

def save_visualizations(vis_data, epoch, vis_dir, max_depth):
    """Save 5-panel PNGs: GT RGB | GT Depth | Pred Depth | Spectrogram | FOA."""
    epoch_dir = os.path.join(vis_dir, f'epoch_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)

    for i, item in enumerate(vis_data):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        gt_rgb = item['gt_rgb']
        if gt_rgb is not None:
            disp = (gt_rgb * 255).astype(np.uint8) if gt_rgb.max() <= 1.0 and gt_rgb.max() > 0 else gt_rgb
            axes[0].imshow(disp)
        else:
            axes[0].text(0.5, 0.5, 'No RGB', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('GT RGB'); axes[0].axis('off')

        gt_d = item['gt_depth']
        if gt_d.ndim == 3 and gt_d.shape[0] == 1:
            gt_d = gt_d[0]
        im1 = axes[1].imshow(np.ma.masked_where(gt_d <= 0, gt_d), cmap='plasma', vmin=0, vmax=max_depth)
        axes[1].set_title('GT Depth'); axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        pred_d = item['pred_depth']
        if pred_d.ndim == 3 and pred_d.shape[0] == 1:
            pred_d = pred_d[0]
        im2 = axes[2].imshow(pred_d, cmap='plasma', vmin=0, vmax=max_depth)
        axes[2].set_title('Pred Depth'); axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        inp = item['input']
        spec = np.log10(inp.mean(axis=0) + 1e-10) if inp.ndim == 3 and inp.shape[0] == 2 else np.log10(inp + 1e-10)
        axes[3].imshow(spec, cmap='magma', aspect='auto', origin='lower')
        axes[3].set_title('Spectrogram'); axes[3].axis('off')

        gt_foa = item.get('gt_foa')
        pred_foa = item.get('pred_foa')
        if gt_foa is not None and pred_foa is not None:
            x_pos = np.arange(4)
            axes[4].bar(x_pos - 0.15, gt_foa, 0.3, label='GT', alpha=0.8)
            axes[4].bar(x_pos + 0.15, pred_foa, 0.3, label='Pred', alpha=0.8)
            axes[4].set_xticks(x_pos); axes[4].set_xticklabels(['W', 'Y', 'Z', 'X'])
            axes[4].legend(); axes[4].set_title('FOA Coeffs')
        else:
            axes[4].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f'sample_{i:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Model builder
# ============================================================

def build_model(cfg):
    num_downs = 7 if cfg.model.generator == 'unet_128' else 8
    return AudioDepthFOAV2Generator(
        cfg, input_nc=6, output_nc=1, num_downs=num_downs, ngf=64,
        use_dropout=False,
        proj_dim=cfg.model.proj_dim,
        foa_dim=cfg.model.foa_dim,
        sh_order=cfg.model.sh_order,
        scale_shift_hidden=cfg.model.scale_shift_hidden,
        scale_shift_layers=cfg.model.scale_shift_layers,
        H_erp=int(cfg.dataset.images_size[0]),
        W_erp=int(cfg.dataset.images_size[1]),
    )


def build_criterion(cfg, device):
    depth_criterion = DepthLoss(
        use_l1=cfg.mode.use_l1, use_silog=cfg.mode.use_silog,
        use_gradient=cfg.mode.use_gradient, use_berhu=cfg.mode.use_berhu,
        use_ssim=cfg.mode.use_ssim,
        w_l1=cfg.mode.w_l1, w_silog=cfg.mode.w_silog,
        w_gradient=cfg.mode.w_gradient, w_berhu=cfg.mode.w_berhu,
        w_ssim=cfg.mode.w_ssim,
    ).to(device)

    foa_criterion = FOAGuidedLoss(
        use_cosine=cfg.model.foa_use_cosine,
        cosine_weight=cfg.model.foa_cosine_weight,
    ).to(device)

    hist_weight = cfg.model.hist_weight
    hist_criterion = None
    if hist_weight > 0:
        hist_criterion = SH5HistogramAlignmentLoss().to(device)

    return AudioDepthFOAV2Loss(
        depth_criterion=depth_criterion,
        foa_criterion=foa_criterion,
        depth_weight=cfg.model.depth_weight,
        foa_weight=cfg.model.foa_weight,
        hist_criterion=hist_criterion,
        hist_weight=hist_weight,
        latent_reg_weight=cfg.model.latent_reg_weight,
    ).to(device)


# ============================================================
# Training
# ============================================================

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print(f"{n_GPU} {device} device(s)")

    batch_size = cfg.mode.batch_size
    sh_order = cfg.model.sh_order
    sh_dim = (sh_order + 1) ** 2
    print(f"SH order: {sh_order} ({sh_dim} coefficients, 4 guided + {sh_dim - 4} learned)")

    # Dataset
    train_set, train_loader = make_dataloader(cfg, 'train', batch_size=batch_size)
    val_set, val_loader = make_dataloader(cfg, 'val', batch_size=batch_size)
    print(f'Train: {len(train_set)} samples, Val: {len(val_set)} samples')

    # Model
    model = build_model(cfg)
    if n_GPU > 0:
        model = model.cuda()
        model = nn.DataParallel(model, list(range(n_GPU)))
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {cfg.model.name} ({total_params:.1f}M params)')

    # Loss & optimizer
    criterion = build_criterion(cfg, device)
    use_hist_align = cfg.model.hist_weight > 0

    lr = cfg.mode.learning_rate
    if cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)

    # Output directories
    project_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{batch_size}_"
                       f"Lr{lr}_{cfg.mode.optimizer}_{cfg.mode.experiment_name}")
    ckpt_dir = os.path.join(project_dir, 'checkpoints', experiment_name)
    vis_dir = os.path.join(project_dir, 'outputs', experiment_name, 'visualizations')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f'Checkpoints: {ckpt_dir}')

    # Resume
    start_epoch = 1
    if cfg.mode.checkpoints is not None:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{cfg.mode.checkpoints}.pth')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    # FOA freeze warmup
    foa_freeze_epochs = cfg.model.foa_freeze_epochs
    def set_sh_branch_frozen(frozen):
        base = get_base_model(model)
        for name in ('audio_proj', 'foa_head', 'hoa_head', 'scale_shift'):
            module = getattr(base, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = not frozen

    best_abs_rel = float('inf')
    n_vis = min(20, len(val_set))
    vis_indices = set(np.linspace(0, len(val_set) - 1, n_vis, dtype=int).tolist())

    dataset_dir = cfg.dataset.dataset_dir
    depth_type = cfg.dataset.depth_type
    target_h, target_w = int(cfg.dataset.images_size[0]), int(cfg.dataset.images_size[1])

    training_start = time.time()
    for epoch in range(start_epoch, cfg.mode.epochs + 1):
        foa_frozen = foa_freeze_epochs > 0 and epoch <= foa_freeze_epochs
        if foa_freeze_epochs > 0:
            set_sh_branch_frozen(foa_frozen)
            if epoch == 1:
                print(f'  [Warmup] SH branch FROZEN for {foa_freeze_epochs} epochs')
            elif epoch == foa_freeze_epochs + 1:
                print(f'  [Warmup done] SH branch UNFROZEN')

        use_hist = use_hist_align and not foa_frozen
        t0 = time.time()
        losses_accum = {'total': [], 'depth': [], 'foa': [], 'hist': []}

        # --- Train ---
        model.train()
        for i, (audio, gtdepth, ambi) in enumerate(train_loader):
            audio, gtdepth, ambi = audio.to(device), gtdepth.to(device), ambi.to(device)
            gt_foa = extract_foa_target_from_energy_map(ambi)

            optimizer.zero_grad()
            outputs = model(audio, return_hist_maps=use_hist, ambi_maps=ambi)

            gt_depth_sh_map, gt_depth_sh_coeffs = None, None
            if use_hist:
                gt_depth_sh_map, gt_depth_sh_coeffs = compute_gt_depth_sh(model, gtdepth)

            if foa_frozen:
                loss = criterion.depth_criterion(outputs["pred_depth"], gtdepth) * criterion.depth_weight
                losses_accum['total'].append(loss.item())
                losses_accum['depth'].append(loss.item() / criterion.depth_weight)
            else:
                loss_dict = criterion(outputs, gtdepth, gt_foa,
                                      gt_depth_sh=gt_depth_sh_map,
                                      gt_depth_sh_coeffs=gt_depth_sh_coeffs)
                loss = loss_dict["total"]
                losses_accum['total'].append(loss.item())
                losses_accum['depth'].append(loss_dict["depth"].item())
                losses_accum['foa'].append(loss_dict["foa"].item())
                if use_hist:
                    losses_accum['hist'].append(loss_dict["hist_align"].item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_batches = len(train_loader)
            if (i + 1) % max(1, total_batches // 5) == 0 or (i + 1) == total_batches:
                progress = (i + 1) / total_batches * 100
                msg = (f'  Epoch {epoch} [{i+1}/{total_batches} {progress:.0f}%] '
                       f'Loss: {np.mean(losses_accum["total"]):.4f} '
                       f'D:{np.mean(losses_accum["depth"]):.4f}')
                if losses_accum['foa']:
                    msg += f' F:{np.mean(losses_accum["foa"]):.4f}'
                if losses_accum['hist']:
                    msg += f' H:{np.mean(losses_accum["hist"]):.4f}'
                print(msg)

        scheduler.step()
        epoch_time = time.time() - t0
        print(f'Epoch [{epoch}/{cfg.mode.epochs}] Loss: {np.mean(losses_accum["total"]):.4f} '
              f'Time: {epoch_time:.1f}s LR: {scheduler.get_last_lr()[0]:.6f}')

        # --- Validation ---
        if epoch % cfg.mode.validation_iter == 0:
            model.eval()
            errors = []
            val_losses = []
            vis_data = []

            with torch.no_grad():
                for batch_idx, (audio_v, gtdepth_v, ambi_v) in enumerate(val_loader):
                    audio_v, gtdepth_v, ambi_v = audio_v.to(device), gtdepth_v.to(device), ambi_v.to(device)
                    gt_foa_v = extract_foa_target_from_energy_map(ambi_v)
                    outputs_v = model(audio_v, return_hist_maps=use_hist, ambi_maps=ambi_v)

                    gt_dsh, gt_dsh_c = None, None
                    if use_hist:
                        gt_dsh, gt_dsh_c = compute_gt_depth_sh(model, gtdepth_v)

                    if foa_frozen:
                        lv = criterion.depth_criterion(outputs_v["pred_depth"], gtdepth_v) * criterion.depth_weight
                    else:
                        lv_dict = criterion(outputs_v, gtdepth_v, gt_foa_v, gt_depth_sh=gt_dsh, gt_depth_sh_coeffs=gt_dsh_c)
                        lv = lv_dict["total"]
                    val_losses.append(lv.item())

                    depth_pred_v = outputs_v["pred_depth"]
                    pred_foa_v = outputs_v["pred_foa"]

                    for idx in range(depth_pred_v.shape[0]):
                        dataset_idx = batch_idx * batch_size + idx
                        if dataset_idx >= len(val_set.samples):
                            break
                        if cfg.dataset.depth_norm:
                            ug = gtdepth_v[idx].cpu().numpy() * cfg.dataset.max_depth
                            up = depth_pred_v[idx].cpu().numpy() * cfg.dataset.max_depth
                        else:
                            ug = gtdepth_v[idx].cpu().numpy()
                            up = depth_pred_v[idx].cpu().numpy()
                        errors.append(compute_errors(ug, up))

                        if dataset_idx in vis_indices:
                            scene_id, step_idx = val_set.samples[dataset_idx]
                            gt_rgb = load_gt_rgb(dataset_dir, scene_id, step_idx, depth_type, target_h, target_w)
                            vis_data.append({
                                'gt_rgb': gt_rgb, 'gt_depth': ug, 'pred_depth': up,
                                'input': audio_v[idx].cpu().numpy(),
                                'gt_foa': gt_foa_v[idx].cpu().numpy(),
                                'pred_foa': pred_foa_v[idx].cpu().numpy(),
                            })

            mean_errors = np.array(errors).mean(0)
            abs_rel = mean_errors[0]
            print(f'  Val Loss: {np.mean(val_losses):.4f} | '
                  f'ABS_REL: {abs_rel:.4f} RMSE: {mean_errors[1]:.4f} '
                  f'd1: {mean_errors[2]:.4f} d2: {mean_errors[3]:.4f} d3: {mean_errors[4]:.4f}')

            if vis_data:
                save_visualizations(vis_data, epoch, vis_dir, cfg.dataset.max_depth)
                print(f'  Saved {len(vis_data)} visualizations')

            if abs_rel < best_abs_rel:
                best_abs_rel = abs_rel
                torch.save({
                    'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'best_abs_rel': best_abs_rel,
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                print(f'  >> Best model saved (ABS_REL: {best_abs_rel:.4f})')

        # Time budget check
        elapsed = time.time() - training_start
        if elapsed >= TIME_BUDGET:
            print(f'\nTime budget reached ({elapsed:.1f}s >= {TIME_BUDGET}s). Stopping.')
            break

    total_time = time.time() - training_start
    print(f'\nTraining complete. Best ABS_REL: {best_abs_rel:.4f}')
    print(f'training_seconds: {total_time:.1f}')


# ============================================================
# Testing
# ============================================================

def test(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print(f"{n_GPU} {device} device(s)")

    batch_size = cfg.mode.batch_size
    eval_on = cfg.mode.eval_on

    eval_set, eval_loader = make_dataloader(cfg, eval_on, batch_size=batch_size)
    print(f'Eval [{eval_on}]: {len(eval_set)} samples')

    # Model
    model = build_model(cfg)
    if n_GPU > 0:
        model = model.cuda()
        model = nn.DataParallel(model, list(range(n_GPU)))

    # Load checkpoint
    project_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_name = (f"{cfg.model.generator}_{cfg.dataset.name}_BS{cfg.mode.batch_size}_"
                       f"Lr{cfg.mode.learning_rate}_{cfg.mode.optimizer}_{cfg.mode.experiment_name}")
    ckpt_dir = os.path.join(project_dir, 'checkpoints', experiment_name)

    load_epoch = cfg.mode.checkpoints
    if load_epoch is None or str(load_epoch) == 'best':
        import glob
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    else:
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{load_epoch}.pth')
    print(f'Loading: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError:
        new_sd = {(k[len('module.'):] if k.startswith('module.') else k): v
                  for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(new_sd)
    print(f'Loaded epoch {ckpt["epoch"]} (ABS_REL: {ckpt.get("best_abs_rel", "N/A")})')

    # Evaluate
    model.eval()
    max_depth = cfg.dataset.max_depth
    dataset_dir = cfg.dataset.dataset_dir
    depth_type = cfg.dataset.depth_type
    target_h, target_w = int(cfg.dataset.images_size[0]), int(cfg.dataset.images_size[1])
    vis_every = cfg.mode.vis_every
    vis_dir = os.path.join(project_dir, 'outputs', experiment_name, 'visualizations', eval_on)
    if vis_every > 0:
        os.makedirs(vis_dir, exist_ok=True)

    depth_errors = []
    foa_errors_list = []
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (audio, depthgt, ambi) in enumerate(eval_loader):
            audio, depthgt, ambi = audio.to(device), depthgt.to(device), ambi.to(device)
            gt_foa_batch = extract_foa_target_from_energy_map(ambi)
            outputs = model(audio, ambi_maps=ambi)
            depth_pred = outputs["pred_depth"]
            pred_foa_batch = outputs["pred_foa"]

            for idx in range(depth_pred.shape[0]):
                dataset_idx = batch_idx * batch_size + idx
                if dataset_idx >= len(eval_set.samples):
                    break
                scene_id, step_idx = eval_set.samples[dataset_idx]

                if cfg.dataset.depth_norm:
                    ug = depthgt[idx].cpu().numpy() * max_depth
                    up = depth_pred[idx].cpu().numpy() * max_depth
                else:
                    ug = depthgt[idx].cpu().numpy()
                    up = depth_pred[idx].cpu().numpy()

                depth_errors.append(compute_errors(ug, up))
                foa_errors_list.append(compute_foa_errors(
                    gt_foa_batch[idx].cpu().numpy(), pred_foa_batch[idx].cpu().numpy()))

                if vis_every > 0 and dataset_idx % vis_every == 0:
                    gt_2d = ug[0] if ug.ndim == 3 and ug.shape[0] == 1 else ug
                    pred_2d = up[0] if up.ndim == 3 and up.shape[0] == 1 else up
                    gt_rgb = load_gt_rgb(dataset_dir, scene_id, step_idx, depth_type, target_h, target_w)
                    inp = audio[idx].cpu().numpy()
                    spec = np.log10(inp.mean(axis=0) + 1e-10) if inp.ndim == 3 and inp.shape[0] == 2 else np.log10(inp + 1e-10)

                    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                    if gt_rgb is not None:
                        axes[0].imshow((gt_rgb * 255).astype(np.uint8))
                    else:
                        axes[0].text(0.5, 0.5, 'No RGB', ha='center', va='center', transform=axes[0].transAxes)
                    axes[0].set_title('GT RGB'); axes[0].axis('off')
                    im1 = axes[1].imshow(np.ma.masked_where(gt_2d <= 0, gt_2d), cmap='plasma', vmin=0, vmax=max_depth)
                    axes[1].set_title('GT Depth'); axes[1].axis('off'); plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                    im2 = axes[2].imshow(pred_2d, cmap='plasma', vmin=0, vmax=max_depth)
                    axes[2].set_title('Pred Depth'); axes[2].axis('off'); plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                    axes[3].imshow(spec, cmap='magma', aspect='auto', origin='lower'); axes[3].set_title('Spectrogram'); axes[3].axis('off')
                    gt_f, pred_f = gt_foa_batch[idx].cpu().numpy(), pred_foa_batch[idx].cpu().numpy()
                    x_pos = np.arange(4)
                    axes[4].bar(x_pos - 0.15, gt_f, 0.3, label='GT', alpha=0.8)
                    axes[4].bar(x_pos + 0.15, pred_f, 0.3, label='Pred', alpha=0.8)
                    axes[4].set_xticks(x_pos); axes[4].set_xticklabels(['W', 'Y', 'Z', 'X']); axes[4].legend()
                    axes[4].set_title('FOA Coeffs')
                    plt.suptitle(f'{scene_id} | idx={step_idx} | sample {dataset_idx}', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'vis_{dataset_idx:05d}_{scene_id}_{step_idx}.png'),
                                dpi=150, bbox_inches='tight')
                    plt.close()
                    vis_count += 1

            if (batch_idx + 1) % 10 == 0:
                total = min((batch_idx + 1) * batch_size, len(eval_set))
                print(f'Processed {batch_idx + 1}/{len(eval_loader)} batches ({total} samples)')

    # Print results
    de = np.array(depth_errors)
    md = de.mean(0)
    foa_l1 = np.mean([e['foa_l1'] for e in foa_errors_list])
    foa_cos = np.mean([e['foa_cosine'] for e in foa_errors_list])
    foa_dir = np.mean([e['foa_dir_cosine'] for e in foa_errors_list])

    print('\n' + '=' * 60)
    print('Test Results — Depth Metrics')
    print('=' * 60)
    print(f'ABS_REL: {md[0]:.4f}')
    print(f'RMSE:    {md[1]:.4f}')
    print(f'Delta1:  {md[2]:.4f}')
    print(f'Delta2:  {md[3]:.4f}')
    print(f'Delta3:  {md[4]:.4f}')
    print(f'Log10:   {md[5]:.4f}')
    print(f'MAE:     {md[6]:.4f}')
    print('=' * 60)
    print(f'Test Results — FOA Metrics (guided channels)')
    print('=' * 60)
    print(f'FOA L1:          {foa_l1:.4f}')
    print(f'FOA Cosine:      {foa_cos:.4f}')
    print(f'FOA Dir Cosine:  {foa_dir:.4f}')
    print('=' * 60)

    if vis_count > 0:
        print(f'{vis_count} visualizations saved to: {vis_dir}')

    # Save stats
    import pandas as pd
    stats = pd.DataFrame({
        'abs_rel': de[:, 0], 'rmse': de[:, 1], 'delta1': de[:, 2],
        'delta2': de[:, 3], 'delta3': de[:, 4], 'log10': de[:, 5], 'mae': de[:, 6],
        'foa_l1': [e['foa_l1'] for e in foa_errors_list],
        'foa_cosine': [e['foa_cosine'] for e in foa_errors_list],
        'foa_dir_cosine': [e['foa_dir_cosine'] for e in foa_errors_list],
    })
    stats_dir = os.path.join(project_dir, 'outputs', experiment_name, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f'stats_{eval_on}.pkl')
    stats.to_pickle(stats_path)
    print(f'Statistics saved to: {stats_path}')


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='AudioDepthFOA V2: Depth from Echoes')

    # Mode
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--eval-on', type=str, default='test', choices=['test', 'val'])

    # Data
    p.add_argument('--dataset-dir', type=str,
                   default='/home/rvi-lab/workspace/sound-spaces/dataset_simplified',
                   help='Path to SoundSpaces dataset')

    # Training
    p.add_argument('--batch-size', type=int, default=32) #64
    p.add_argument('--epochs', type=int, default=40) #80
    p.add_argument('--lr', type=float, default=0.0025)
    p.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam', 'SGD'])
    p.add_argument('--num-workers', type=int, default=16)#32
    p.add_argument('--foa-freeze-epochs', type=int, default=0,
                   help='Depth-only warmup epochs before enabling SH branch') #20 /10

    # Experiment
    p.add_argument('--experiment-name', type=str, default='audio_depth_foa_v2_echoes_erp_v1')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Checkpoint to resume (epoch number or "best")')
    p.add_argument('--vis-every', type=int, default=100,
                   help='Visualize every N samples during test (0=skip)') #100

    return p.parse_args()


if __name__ == '__main__':
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Performance tuning
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')

    args = parse_args()
    cfg = make_config(args)

    print('=' * 60)
    print(f'AudioDepthFOA V2 — mode={args.mode}')
    print(f'Dataset: {args.dataset_dir}')
    print(f'Batch size: {args.batch_size}, LR: {args.lr}, Optimizer: {args.optimizer}')
    print('=' * 60)

    if args.mode == 'train':
        train(cfg)
    elif args.mode == 'test':
        test(cfg)
