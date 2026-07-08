"""
Data preparation and evaluation utilities for the RayDPT audio->ERP-depth model.

Ported from the `test_for_audio_implicit_full` repo (RayDPT pipeline) into the
two-file autoresearch layout. This file is the FIXED data + evaluation harness:

    - erp_grid / sh_basis_matrix : ERP spherical geometry (shared with the model's
                                   per-ray feature bank in train.py)
    - get_scene_split            : deterministic train/val/test scene split
    - SoundSpacesDataset         : binaural audio -> {spec, depth, mask}
    - make_dataloader            : DataLoader factory (dict-batch collate)
    - compute_errors             : depth error metrics (the ground-truth metric)
    - load_gt_rgb                : visualization helper

Frame convention (matches RayDPT's RayBank):
    x = front, y = left(+)/right(-) ear axis, z = up.
    az in (-pi, pi],  el in (-pi/2, pi/2),  cell-centred.
    dir = (cos el cos az, cos el sin az, sin el)

Usage:
    from prepare import SoundSpacesDataset, make_dataloader, compute_errors, erp_grid
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
# ERP spherical geometry (shared by the model's ray-feature bank)
# ============================================================

def erp_grid(H, W):
    """Cell-centred elevation/azimuth grids for an HxW equirectangular map.

    Returns (el, az), each (H, W) float32.
        az in (-pi, pi]  increasing left->right
        el in (pi/2, -pi/2) increasing top->bottom
    """
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    az = (jj + 0.5) / W * 2.0 * np.pi - np.pi
    el = np.pi / 2.0 - (ii + 0.5) / H * np.pi
    return el.astype(np.float32), az.astype(np.float32)


def radial_factor(H, W):
    """Per-pixel factor that converts cubemap perspective (perpendicular-Z) ERP
    depth -> radial (along-ray) depth: r = d / max(|dx|, |dy|, |dz|).

    The SoundSpaces `erp_depth` maps store the per-face perpendicular distance
    (stitched cubemap), so a flat wall back-projects as a circular arc unless
    rescaled. Dividing by the dominant ray-direction axis recovers true radial
    depth (verified: flat floor/ceiling, vertical walls). RayDPT trains on radial
    depth (the source repo's `erp_depth_radial`), so we apply this on the fly.
    """
    el, az = erp_grid(H, W)
    dx = np.cos(el) * np.cos(az)
    dy = np.cos(el) * np.sin(az)
    dz = np.sin(el)
    linf = np.maximum.reduce([np.abs(dx), np.abs(dy), np.abs(dz)])
    return (1.0 / np.clip(linf, 1e-6, None)).astype(np.float32)        # (H, W)


# ------------------------------------------------------------
# Real spherical-harmonic basis (ACN / SN3D), used by the optional
# SH ray-PE in the model and available as a fixed geometry primitive.
# ------------------------------------------------------------

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
    """SH basis for a grid. elevation/azimuth (H,W) or (N,). Returns (N, (L+1)^2)."""
    n_ch = (max_order + 1) ** 2
    el_flat = np.asarray(elevation).ravel()
    az_flat = np.asarray(azimuth).ravel()
    B = np.zeros((el_flat.size, n_ch), dtype=np.float64)
    for q in range(n_ch):
        B[:, q] = _real_sh_sn3d_np(q, el_flat, az_flat)
    return B


# ============================================================
# Scene splitting (deterministic; no scene_split.json required)
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
# Dataset: binaural audio -> ERP radial depth
# ============================================================

_C = 340.0                       # speed of sound [m/s]
_NFFT, _HOP, _WIN = 512, 160, 400

# ------------------------------------------------------------------------------------------
# PROPOSAL-01 acoustic-representation seam (research-editable; benchmark stays fixed).
#
# FIXED / reproducibility-critical (never change these to keep E0-E134 reproducible):
#   get_scene_split (data split), SoundSpacesDataset._wave (waveform access), ._depth (target),
#   compute_errors (metric), swap_audio_lr (L/R symmetry). These define the benchmark.
#
# EDITABLE research logic: the waveform -> input-feature mapping. Set `prepare.FEATURE_FN` from
# train.py to research alternative acoustic representations (multi-resolution STFT, early/late echo
# split, cross-channel coherence, ...). It receives the FIXED raw waveform and the dataset instance:
#     def FEATURE_FN(wav, ds) -> Tensor (in_ch, ds.H, ds.W)
# and may reuse ds._specN / ds._spec2 as building blocks. When FEATURE_FN is None (default) the
# item pipeline is BYTE-IDENTICAL to the pre-refactor baseline (the 5ch [logL,logR,ILD,cosIPD,sinIPD]
# representation), so no historical result is invalidated.
FEATURE_FN = None


class SoundSpacesDataset(Dataset):
    """Binaural echoes -> ERP depth, at cfg.dataset.images_size.

    Each item is a dict:
        spec  : (in_ch, H, W)  log-mag binaural spectrogram (in_ch=2) or RIR
                spatial feature [logL, logR, ILD, cosIPD, sinIPD][:in_ch] (in_ch in {3,5})
        depth : (1, H, W)      radial depth / max_depth in [0, 1]
        mask  : (1, H, W)      valid-depth mask in {0, 1}
        key   : "<scene>/<idx>"
    """

    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        d = cfg.dataset
        self.root_dir = d.dataset_dir
        self.depth_type = d.depth_type
        self.max_depth = d.max_depth
        self.H, self.W = int(d.images_size[0]), int(d.images_size[1])
        self.in_ch = int(getattr(d, 'in_ch', 2))
        self.sr = int(getattr(d, 'sample_rate', 48000))
        self.log_spec = bool(getattr(d, 'log_spec', True))
        self.win_m = float(getattr(d, 'audio_window_m', 0) or self.max_depth)
        self.cut = int(2.0 * self.win_m / _C * self.sr)
        # cubemap perpendicular-Z -> radial depth conversion (see radial_factor)
        self._radial = torch.from_numpy(radial_factor(self.H, self.W))[None]   # (1,H,W)

        scene_split = get_scene_split(self.root_dir, d.split_ratio, seed=d.split_seed)
        self.scenes = scene_split[split]

        self.spectro = T.Spectrogram(n_fft=_NFFT, win_length=_WIN,
                                     hop_length=_HOP, power=1.0)
        self._win = torch.hann_window(_WIN)

        self.samples = []
        for scene in self.scenes:
            audio_dir = os.path.join(self.root_dir, scene, 'audio_wav')
            depth_dir = os.path.join(self.root_dir, scene, f'{self.depth_type}_depth')
            if not (os.path.isdir(audio_dir) and os.path.isdir(depth_dir)):
                continue
            for af in sorted(f for f in os.listdir(audio_dir) if f.endswith('.wav')):
                idx = af.replace('audio_', '').replace('.wav', '')
                depth_path = os.path.join(depth_dir, f'{self.depth_type}_depth_{idx}.npy')
                if os.path.exists(depth_path):
                    self.samples.append((scene, idx))

        print(f"[{split}] {len(self.samples)} samples from {len(self.scenes)} scenes "
              f"({self.H}x{self.W}, in_ch={self.in_ch})", flush=True)

    def __len__(self):
        return len(self.samples)

    def _wave(self, scene, idx):
        wav, sr = torchaudio.load(
            os.path.join(self.root_dir, scene, 'audio_wav', f'audio_{idx}.wav'))
        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)
        wav = wav[:, :self.cut]
        if wav.shape[1] < self.cut:
            wav = F.pad(wav, (0, self.cut - wav.shape[1]))
        return wav

    def _spec2(self, wav):
        """2ch binaural magnitude spectrogram (log1p optional), resized to (H,W)."""
        sp = self.spectro(wav)                                 # (2,F,T')
        if self.log_spec:
            sp = torch.log1p(sp)
        return F.interpolate(sp.unsqueeze(0), (self.H, self.W),
                             mode='nearest').squeeze(0).float()

    def _specN(self, wav, n):
        """RIR spatial feature [logL, logR, ILD, cosIPD, sinIPD][:n], resized to (H,W)."""
        st = torch.stft(wav, _NFFT, _HOP, _WIN, self._win, return_complex=True)  # (2,F,T')
        L, R = st[0], st[1]
        eps = 1e-6
        lmag = torch.log1p(L.abs()); rmag = torch.log1p(R.abs())
        ild = torch.log(L.abs() + eps) - torch.log(R.abs() + eps)
        ipd = torch.angle(L * torch.conj(R))
        feat = torch.stack([lmag, rmag, ild, torch.cos(ipd), torch.sin(ipd)], 0)[:n]
        return F.interpolate(feat.unsqueeze(0), (self.H, self.W),
                             mode='nearest').squeeze(0).float()

    def _depth(self, scene, idx):
        d = np.nan_to_num(np.load(os.path.join(
            self.root_dir, scene, f'{self.depth_type}_depth',
            f'{self.depth_type}_depth_{idx}.npy')).astype(np.float32))
        d[d < 0] = 0.0
        t = F.interpolate(torch.from_numpy(d)[None, None], (self.H, self.W),
                          mode='nearest').squeeze(0)            # (1,H,W) perspective-Z
        t = t * self._radial                                   # -> radial (along-ray) depth
        t = t.clamp(max=self.max_depth)
        return t / self.max_depth, (t > 0).float()

    def __getitem__(self, i):
        scene, idx = self.samples[i]
        try:
            wav = self._wave(scene, idx)                # FIXED waveform access
            if FEATURE_FN is not None:                  # PROPOSAL-01 research representation
                spec = FEATURE_FN(wav, self)
            else:                                       # default: byte-identical baseline
                spec = self._spec2(wav) if self.in_ch == 2 else self._specN(wav, self.in_ch)
            depth, mask = self._depth(scene, idx)
        except Exception as e:
            print(f"[skip {scene}/{idx}] {e}", flush=True)
            return self[(i + 1) % len(self)]
        return {"spec": spec.contiguous(), "depth": depth.contiguous(),
                "mask": mask.contiguous(), "key": f"{scene}/{idx}"}


def collate(batch):
    return {k: ([x[k] for x in batch] if k == "key"
                else torch.stack([x[k] for x in batch]))
            for k in batch[0]}


def swap_audio_lr(spec):
    """Channel-aware binaural L/R swap (the physically-correct mirror for this rig).

    Negates the L-R-antisymmetric channels (ILD, sin(IPD)); cos(IPD) is symmetric.
        2ch [L,R]                         -> [R,L]
        3ch [Lmag,Rmag,ILD]               -> [Rmag,Lmag,-ILD]
        5ch [Lmag,Rmag,ILD,cosIPD,sinIPD] -> [Rmag,Lmag,-ILD,cosIPD,-sinIPD]
    Pairs with a width-flip (az -> -az) of depth/mask for L/R mirror augmentation.
    """
    C = spec.shape[1]
    if C == 2:
        return spec[:, [1, 0]]
    y = spec.clone()
    y[:, 0], y[:, 1] = spec[:, 1], spec[:, 0]
    if C >= 3:
        y[:, 2] = -spec[:, 2]                       # ILD -> -ILD
    if C >= 5:
        y[:, 3] = spec[:, 3]; y[:, 4] = -spec[:, 4]  # cos(IPD) same, sin(IPD) -> -sin(IPD)
    return y


def make_dataloader(cfg, split, batch_size=None, shuffle=None):
    """Create (dataset, dataloader) for the given split."""
    dataset = SoundSpacesDataset(cfg, split=split)
    if batch_size is None:
        batch_size = cfg.mode.batch_size
    if shuffle is None:
        shuffle = (split == 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=cfg.mode.num_threads, collate_fn=collate,
                        pin_memory=True, drop_last=shuffle)
    return dataset, loader


# ============================================================
# Evaluation metrics (ground-truth metric — do not weaken)
# ============================================================

def compute_errors(gt, pred):
    """Depth error metrics between predicted and ground-truth depth (in METRES).

    Masks invalid (gt <= 0) pixels. Returns:
        (abs_rel, rmse, a1, a2, a3, log_10, mae)
    """
    gt = np.asarray(gt); pred = np.asarray(pred)
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

    out = [abs_rel, rmse, a1, a2, a3, log_10, mae]
    return tuple(0.0 if (v != v) else float(v) for v in out)


# ============================================================
# Visualization helper
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
