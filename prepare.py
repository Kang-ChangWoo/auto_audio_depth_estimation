"""
Data preparation and evaluation utilities for AudioDepthFOA V2.

Provides:
    - SoundSpacesDataset: binaural echoes -> ERP depth dataset
    - compute_errors: depth error metrics
    - compute_foa_errors: FOA evaluation metrics
    - get_scene_split: deterministic train/val/test scene splitting

Usage:
    from prepare import SoundSpacesDataset, compute_errors, compute_foa_errors
"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T

from scipy.special import lpmv
from scipy.special import factorial as sp_factorial
from PIL import Image


# ============================================================
# SH basis computation (ACN / SN3D) — needed by dataset
# ============================================================

def _acn_to_nm(acn):
    n = int(math.floor(math.sqrt(acn)))
    m = acn - n * n - n
    return n, m


def _sn3d_norm(n, m):
    m_abs = abs(m)
    delta = 1.0 if m == 0 else 0.0
    return math.sqrt((2.0 - delta) * sp_factorial(n - m_abs, exact=True)
                     / sp_factorial(n + m_abs, exact=True))


def _real_sh_sn3d_np(acn, elevation, azimuth):
    n, m = _acn_to_nm(acn)
    m_abs = abs(m)
    N = _sn3d_norm(n, m)
    P = (-1)**m_abs * lpmv(m_abs, n, np.sin(elevation))
    if m > 0:
        return N * P * np.cos(m * azimuth)
    elif m == 0:
        return N * P
    else:
        return N * P * np.sin(m_abs * azimuth)


def sh_basis_matrix(max_order, elevation, azimuth):
    """Compute SH basis matrix for given grid. Returns (N_pixels, N_channels)."""
    n_ch = (max_order + 1) ** 2
    el_flat = elevation.ravel()
    az_flat = azimuth.ravel()
    B = np.zeros((el_flat.size, n_ch))
    for q in range(n_ch):
        B[:, q] = _real_sh_sn3d_np(q, el_flat, az_flat)
    return B


def reconstruct_per_component_maps(sh_coeffs, B):
    """Reconstruct per-SH-component energy maps.
    Args:
        sh_coeffs: (n_ch, T) SH time-domain signals
        B: (N_pixels, n_ch) precomputed SH basis matrix
    Returns:
        (n_ch, N_pixels) per-component spatial energy maps
    """
    n_ch = B.shape[1]
    A = sh_coeffs[:n_ch]
    rms = np.sqrt(np.mean(A ** 2, axis=1))
    maps = B * rms[None, :]
    return maps.T


# ============================================================
# Scene splitting
# ============================================================

def get_scene_split(dataset_dir, split_ratio, seed=42):
    scenes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    n = len(scenes)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    split = {
        'train': sorted(scenes[:n_train]),
        'val': sorted(scenes[n_train:n_train + n_val]),
        'test': sorted(scenes[n_train + n_val:]),
    }
    print(f"Scene split — train: {len(split['train'])}, "
          f"val: {len(split['val'])}, test: {len(split['test'])}")
    return split


# ============================================================
# Dataset
# ============================================================

class SoundSpacesDataset(Dataset):
    """Sound-Spaces dataset: binaural echoes -> ERP depth."""

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.root_dir = cfg.dataset.dataset_dir
        self.audio_format = cfg.dataset.audio_format
        self.depth_type = cfg.dataset.depth_type
        self.max_depth = cfg.dataset.max_depth
        self.min_depth = cfg.dataset.min_depth
        self.use_ambisonic = getattr(cfg.dataset, 'use_ambisonic', False)

        scene_split = get_scene_split(
            self.root_dir, cfg.dataset.split_ratio, seed=cfg.dataset.split_seed)
        self.scenes = scene_split[split]

        self.samples = []
        skipped = 0
        for scene in self.scenes:
            audio_dir = os.path.join(self.root_dir, scene, 'audio_wav')
            depth_dir = os.path.join(self.root_dir, scene, f'{self.depth_type}_depth')
            if not os.path.isdir(audio_dir) or not os.path.isdir(depth_dir):
                continue
            if self.use_ambisonic:
                ambi_dir = os.path.join(self.root_dir, scene, 'ambi1_npy')
                if not os.path.isdir(ambi_dir):
                    continue

            audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
            for af in audio_files:
                idx = af.replace('audio_', '').replace('.wav', '')
                depth_file = f'{self.depth_type}_depth_{idx}.npy'
                depth_path = os.path.join(depth_dir, depth_file)
                if not os.path.exists(depth_path):
                    continue
                if self.use_ambisonic:
                    ambi_path = os.path.join(self.root_dir, scene, 'ambi1_npy', f'ambi1_{idx}.npy')
                    if not os.path.exists(ambi_path):
                        continue
                depth = np.load(depth_path).astype(np.float32)
                if np.mean(depth <= 0) > 0.1:
                    skipped += 1
                    continue
                self.samples.append((scene, idx))

        print(f"[{split}] {len(self.samples)} samples from {len(self.scenes)} scenes "
              f"(filtered {skipped} with >10% no-depth)"
              f"{' [ambisonic=ON]' if self.use_ambisonic else ''}")

        if self.use_ambisonic:
            h, w = cfg.dataset.images_size
            h, w = int(h), int(w)
            jj, ii = np.meshgrid(np.arange(w), np.arange(h))
            az_grid = (jj + 0.5) / w * 2 * np.pi - np.pi
            el_grid = np.pi / 2 - (ii + 0.5) / h * np.pi
            self._sh_basis = sh_basis_matrix(1, el_grid, az_grid)
            self._erp_shape = (h, w)
            self._sh_n_ch = 4
            print(f"  Precomputed SH basis matrix: {self._sh_basis.shape} (order=1)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, sample_idx = self.samples[idx]

        # Load binaural audio
        audio_path = os.path.join(self.root_dir, scene, 'audio_wav', f'audio_{sample_idx}.wav')
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.clone()

        n_fft, hop_length, win_length = 512, 160, 400
        cut = int((2 * 20.0 / 340) * sr)
        waveform = waveform[:, :cut]

        if 'spectrogram' in self.audio_format:
            audio = self._get_spectrogram(waveform, n_fft=n_fft, power=1.0,
                                          win_length=win_length, hop_length=hop_length)
            images_size = self.cfg.dataset.images_size
            target_size = tuple(int(x) for x in images_size)
            audio = F.interpolate(audio.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
        else:
            audio = waveform

        # Load ERP depth
        depth_path = os.path.join(
            self.root_dir, scene, f'{self.depth_type}_depth',
            f'{self.depth_type}_depth_{sample_idx}.npy')
        depth = np.load(depth_path).astype(np.float32)
        depth = np.nan_to_num(depth)
        depth[depth == -np.inf] = 0
        depth[depth == np.inf] = 0
        depth[depth < 0.0] = 0.0
        depth[depth > self.max_depth] = self.max_depth
        gt_depth = torch.from_numpy(depth).unsqueeze(0)

        if 'resize' in self.cfg.dataset.preprocess:
            h, w = self.cfg.dataset.images_size
            gt_depth = F.interpolate(gt_depth.unsqueeze(0), size=(int(h), int(w)),
                                     mode='nearest').squeeze(0)
        if self.cfg.dataset.depth_norm:
            gt_depth = gt_depth / self.max_depth

        if self.use_ambisonic:
            ambi_path = os.path.join(
                self.root_dir, scene, 'ambi1_npy', f'ambi1_{sample_idx}.npy')
            sh_coeffs = np.load(ambi_path).astype(np.float64)
            h, w = self._erp_shape
            component_maps = reconstruct_per_component_maps(sh_coeffs, self._sh_basis)
            component_maps = component_maps.reshape(4, h, w).astype(np.float32)
            for ch in range(4):
                cmax = np.abs(component_maps[ch]).max()
                if cmax > 0:
                    component_maps[ch] = component_maps[ch] / cmax
            ambi_erp = torch.from_numpy(component_maps)
            return audio.contiguous(), gt_depth.contiguous(), ambi_erp.contiguous()

        return audio.contiguous(), gt_depth.contiguous()

    def _get_spectrogram(self, waveform, n_fft=512, power=1.0, win_length=64, hop_length=16):
        spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_length,
                                    power=power, hop_length=hop_length)
        return spectrogram(waveform)


def make_dataloader(cfg, split, batch_size=None, shuffle=None):
    """Create a DataLoader for the given split.
    Args:
        cfg: config object with dataset and mode attributes
        split: 'train', 'val', or 'test'
        batch_size: override cfg.mode.batch_size if provided
        shuffle: override default (True for train, False otherwise)
    Returns:
        (dataset, dataloader) tuple
    """
    dataset = SoundSpacesDataset(cfg, split=split)
    if batch_size is None:
        batch_size = cfg.mode.batch_size
    if shuffle is None:
        shuffle = (split == 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=cfg.mode.num_threads, pin_memory=True)
    return dataset, loader


# ============================================================
# Evaluation metrics
# ============================================================

def compute_errors(gt, pred):
    """Depth error metrics between predicted and ground truth."""
    mask = gt > 0
    pred, gt = pred[mask], gt[mask]
    if len(pred) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = np.abs(np.log10(gt + 1e-8) - np.log10(pred + 1e-8)).mean()
    mae = np.abs(gt - pred).mean()

    for v in [rmse, a1, a2, a3, abs_rel, log_10, mae]:
        if v != v:
            v = 0.0
    return abs_rel, rmse, a1, a2, a3, log_10, mae


def compute_foa_errors(gt_foa, pred_foa):
    """FOA evaluation metrics (guided channels only)."""
    foa_l1 = np.abs(gt_foa - pred_foa).mean()
    dot = np.dot(gt_foa, pred_foa)
    foa_cosine = dot / (np.linalg.norm(gt_foa) + 1e-8) / (np.linalg.norm(pred_foa) + 1e-8)
    gt_dir, pred_dir = gt_foa[1:], pred_foa[1:]
    foa_dir_cosine = np.dot(gt_dir, pred_dir) / (np.linalg.norm(gt_dir) + 1e-8) / (np.linalg.norm(pred_dir) + 1e-8)
    return {'foa_l1': float(foa_l1), 'foa_cosine': float(foa_cosine), 'foa_dir_cosine': float(foa_dir_cosine)}


# ============================================================
# Visualization helpers
# ============================================================

def load_gt_rgb(dataset_dir, scene, sample_idx, depth_type, h=256, w=512):
    if depth_type == 'erp':
        rgb_path = os.path.join(dataset_dir, scene, 'erp_rgb', f'erp_{sample_idx}.png')
    else:
        rgb_path = os.path.join(dataset_dir, scene, 'pinhole_rgb', f'pinhole_{sample_idx}.png')
    if os.path.exists(rgb_path):
        img = Image.open(rgb_path).convert('RGB').resize((w, h), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0
    return None
