"""
Data preparation and evaluation utilities for the RayDPT audio->ERP-depth model.

Depth convention: PLANAR (cubemap perpendicular-Z), i.e. the SoundSpaces `erp_depth`
maps are used as stored, with no along-ray (radial) rescaling.

Ported from the `test_for_audio_implicit_full` repo (RayDPT pipeline) into the
two-file autoresearch layout. This file is the FIXED data + evaluation harness:

    - erp_grid                   : ERP spherical geometry (shared with the model's
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
# Dataset: binaural audio -> ERP planar (cubemap) depth
# ============================================================

_C = 340.0                       # speed of sound [m/s]  -- FIXED (physics; sets the waveform cut)

# ------------------------------------------------------------------------------------------
# RESPONSIBILITY BOUNDARY (see program.md).
#
#   FIXED (these ARE the benchmark; never edit):
#       get_scene_split   split          _wave   waveform access + cut
#       _depth            target         compute_errors   metric
#   Note `_wave`'s cut depends only on (_C, audio_window_m, sample_rate) -- NOT on the STFT
#   parameters below. Changing the analysis window therefore cannot change which samples
#   exist, which audio is read, or what the target is.
#
#   EDITABLE research logic (the acoustic representation):
#       the STFT analysis window, the cue set, and `_features` itself.
#
# STFT defaults reproduce the historical representation EXACTLY (512/160/400). They are now
# read from cfg.dataset so that temporal-resolution research is possible without editing this
# file. Physical meaning of the hop, which is what makes this a research knob and not a
# nuisance parameter: an echo from a surface at depth d arrives at t = 2d/c, so one hop of H
# samples quantises depth at c*H/(2*sr). At the default H=160, sr=48k that is 0.567 m, and the
# 58.8 ms window yields only T=18 time frames -- which `_features` then interpolates up to W.
# ------------------------------------------------------------------------------------------
_NFFT, _HOP, _WIN = 512, 160, 400       # historical defaults; overridable via cfg.dataset


def depth_quantum_m(hop, sample_rate=48000, c=_C):
    """One-way depth resolution implied by an STFT hop, in metres: c*hop/(2*sr)."""
    return c * hop / (2.0 * sample_rate)

# ------------------------------------------------------------------------------------------
# Input acoustic representation (research-editable; the benchmark split/target/metric/waveform
# stay FIXED). The waveform -> input-feature mapping is a set of NAMED binaural cues, each
# independently toggled on/off, plus a `use_log` switch for magnitude compression:
#
#   logL / L   : left-channel  STFT magnitude   (log1p-compressed iff use_log)
#   logR / R   : right-channel STFT magnitude   (log1p-compressed iff use_log)
#   ILD        : interaural level difference     log|L| - log|R|   (always a log-ratio)
#   cosIPD     : cos of interaural phase diff     angle(L * conj(R))
#   sinIPD     : sin of interaural phase diff
#
# Channels are always assembled in this canonical order (filtered by the flags), so the L/R
# mirror augmentation (`swap_audio_lr`) can act on them by name. Defaults = all five cues on,
# use_log=True  ->  the 5ch [logL, logR, ILD, cosIPD, sinIPD] representation.
#
# For advanced research (multi-resolution STFT, early/late split, coherence, ...) set the
# `prepare.FEATURE_FN` override from the training script: FEATURE_FN(wav, ds) -> (C, ds.H, ds.W).
FEATURE_FN = None

# Canonical cue order + which cues are L/R-antisymmetric (negated on an L/R channel swap).
_CUE_ORDER = ['L', 'R', 'ILD', 'cosIPD', 'sinIPD']
_ANTISYM_CUES = {'ILD', 'sinIPD'}       # cosIPD is symmetric; L/R magnitudes are exchanged


def build_channel_names(use_log=True, feat_L=True, feat_R=True, feat_ILD=True,
                        feat_cosIPD=True, feat_sinIPD=True):
    """Ordered channel names for the enabled cues (magnitude names carry a 'log' prefix when
    use_log). Used to size in_ch and to drive the name-aware L/R swap."""
    flags = {'L': feat_L, 'R': feat_R, 'ILD': feat_ILD,
             'cosIPD': feat_cosIPD, 'sinIPD': feat_sinIPD}
    names = []
    for cue in _CUE_ORDER:
        if not flags[cue]:
            continue
        names.append(('log' + cue) if (use_log and cue in ('L', 'R')) else cue)
    return names


class SoundSpacesDataset(Dataset):
    """Binaural echoes -> ERP depth, at cfg.dataset.images_size.

    Each item is a dict:
        spec  : (in_ch, H, W)  binaural cue stack, canonical order filtered by the feat_* flags:
                [logL/L, logR/R, ILD, cosIPD, sinIPD]  (see build_channel_names / _features)
        depth : (1, H, W)      planar (cubemap) depth / max_depth in [0, 1]
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
        self.sr = int(getattr(d, 'sample_rate', 48000))
        # --- input representation: named cue toggles + log switch ---
        self.use_log = bool(getattr(d, 'use_log', True))
        self.feat_flags = dict(
            feat_L=bool(getattr(d, 'feat_L', True)),
            feat_R=bool(getattr(d, 'feat_R', True)),
            feat_ILD=bool(getattr(d, 'feat_ILD', True)),
            feat_cosIPD=bool(getattr(d, 'feat_cosIPD', True)),
            feat_sinIPD=bool(getattr(d, 'feat_sinIPD', True)),
        )
        self.channel_names = build_channel_names(self.use_log, **self.feat_flags)
        if not self.channel_names:
            raise ValueError("No input cues enabled: turn on at least one feat_* flag.")
        self.in_ch = len(self.channel_names)
        self.win_m = float(getattr(d, 'audio_window_m', 0) or self.max_depth)
        self.cut = int(2.0 * self.win_m / _C * self.sr)     # FIXED: independent of the STFT params
        # --- editable acoustic representation: STFT analysis window ---
        self.nfft = int(getattr(d, 'stft_nfft', _NFFT))
        self.hop = int(getattr(d, 'stft_hop', _HOP))
        self.stft_win = int(getattr(d, 'stft_win', _WIN))
        self.feat_interp = str(getattr(d, 'feat_interp', 'nearest'))   # nearest | bilinear

        scene_split = get_scene_split(self.root_dir, d.split_ratio, seed=d.split_seed)
        self.scenes = scene_split[split]

        self._win = torch.hann_window(self.stft_win)

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

        n_frames = 1 + self.cut // self.hop
        print(f"[{split}] {len(self.samples)} samples from {len(self.scenes)} scenes "
              f"({self.H}x{self.W}, in_ch={self.in_ch} [{','.join(self.channel_names)}]) "
              f"stft(nfft={self.nfft},hop={self.hop},win={self.stft_win}) -> {n_frames} time frames, "
              f"depth quantum {depth_quantum_m(self.hop, self.sr):.3f} m", flush=True)

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

    def _features(self, wav):
        """Assemble the enabled binaural cues (canonical order) -> (in_ch, H, W).

        logL/L, logR/R : STFT magnitude (log1p iff use_log); ILD = log|L|-log|R|;
        cosIPD/sinIPD  : cos/sin of the interaural phase difference angle(L*conj(R)).
        """
        st = torch.stft(wav, self.nfft, self.hop, self.stft_win, self._win,
                        return_complex=True)                                     # (2,F,T')
        L, R = st[0], st[1]
        la, ra = L.abs(), R.abs()
        eps = 1e-6
        f = self.feat_flags
        chans = []
        if f['feat_L']:
            chans.append(torch.log1p(la) if self.use_log else la)
        if f['feat_R']:
            chans.append(torch.log1p(ra) if self.use_log else ra)
        if f['feat_ILD']:
            chans.append(torch.log(la + eps) - torch.log(ra + eps))
        if f['feat_cosIPD'] or f['feat_sinIPD']:
            ipd = torch.angle(L * torch.conj(R))
            if f['feat_cosIPD']:
                chans.append(torch.cos(ipd))
            if f['feat_sinIPD']:
                chans.append(torch.sin(ipd))
        feat = torch.stack(chans, 0)                           # (in_ch, F, T')
        # The (F, T') grid is resized to (H, W); the WIDTH axis is STFT TIME, and with the
        # default hop only T'=18 frames are stretched over 512 columns. 'nearest' therefore
        # emits a staircase of ~28 px blocks. 'bilinear' removes the staircase WITHOUT adding
        # any information -- the discriminating control for whether E5's gain was interpolation
        # smoothness rather than temporal information (idea I10).
        kw = {} if self.feat_interp == 'nearest' else {'align_corners': False}
        return F.interpolate(feat.unsqueeze(0), (self.H, self.W),
                             mode=self.feat_interp, **kw).squeeze(0).float()

    def _depth(self, scene, idx):
        d = np.nan_to_num(np.load(os.path.join(
            self.root_dir, scene, f'{self.depth_type}_depth',
            f'{self.depth_type}_depth_{idx}.npy')).astype(np.float32))
        d[d < 0] = 0.0
        t = F.interpolate(torch.from_numpy(d)[None, None], (self.H, self.W),
                          mode='nearest').squeeze(0)            # (1,H,W) planar cubemap depth
        t = t.clamp(max=self.max_depth)
        return t / self.max_depth, (t > 0).float()

    def __getitem__(self, i):
        scene, idx = self.samples[i]
        try:
            wav = self._wave(scene, idx)                # FIXED waveform access
            spec = FEATURE_FN(wav, self) if FEATURE_FN is not None else self._features(wav)
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


def swap_audio_lr(spec, channel_names=None):
    """Channel-aware binaural L/R swap (the physically-correct mirror for this rig).

    Exchanges the L/R magnitude channels and negates the L-R-antisymmetric cues
    (ILD, sinIPD); cosIPD is symmetric. Pairs with a width-flip (az -> -az) of depth/mask
    for L/R mirror augmentation.

    `channel_names` (from SoundSpacesDataset.channel_names) makes the swap work for ANY
    enabled cue subset by name. When None, falls back to the legacy positional convention
    for the classic 2/3/5ch stacks:
        2ch [L,R] -> [R,L]; 3ch [..,ILD] -> [..,-ILD]; 5ch adds cosIPD (same), sinIPD (neg).
    """
    if channel_names is not None:
        name_to_idx = {n: i for i, n in enumerate(channel_names)}
        partner = {'logL': 'logR', 'logR': 'logL', 'L': 'R', 'R': 'L'}
        y = spec.clone()
        for i, n in enumerate(channel_names):
            if n in partner and partner[n] in name_to_idx:
                y[:, i] = spec[:, name_to_idx[partner[n]]]      # exchange L<->R magnitude
            elif n in _ANTISYM_CUES:
                y[:, i] = -spec[:, i]                           # ILD / sinIPD -> negate
            else:
                y[:, i] = spec[:, i]                            # cosIPD (and any solo mag) unchanged
        return y

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
