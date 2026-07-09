#!/usr/bin/env python3
"""
Reporting / visualization for Auto Audio Depth Estimation.

Produces the figures embedded in README.md (written under out/display/):

  qualitative  ->  out/display/qualitative.png
        A grid: rows = a few held-out val scenes, columns =
        [RGB | GT depth | pred(batvision) | pred(best1) | pred(best2)].
        Missing checkpoints render a light "pending" tile, so the intended
        5-column layout is visible from the start and auto-fills as better
        models ("my model", train.py) are trained.

  progress     ->  out/display/score_progress.png
        RMSE, ABS_REL and a1 (d1, δ<1.25) each as a full-width graph stacked
        vertically vs experiment index (from out/results.tsv), running-best
        highlighted. "No experiments yet" placeholder when results.tsv is empty.

  readme       ->  rewrites the <!-- RESULTS:START/END --> block in README.md
        with a compact metrics table generated from out/results.tsv.

  prune        ->  image retention for outputs/**/visualizations (keep the
        earliest "initial" epoch + best/milestone epochs + the latest N;
        delete the rest). See prune_visualizations().

  all          ->  qualitative + progress + readme  (add --prune to also prune)

Usage:
    conda activate ss && python utils/report.py all
    python utils/report.py qualitative
    python utils/report.py prune --keep-latest 6
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DISPLAY_DIR = os.path.join(ROOT, 'out', 'display')
RESULTS_TSV = os.path.join(ROOT, 'out', 'results.tsv')
README = os.path.join(ROOT, 'README.md')
MAX_DEPTH = 10.0


# ------------------------------------------------------------------
# Model registry: label -> (builder module, checkpoint dir). The
# batvision reference is always first; best1/best2 are "my model"
# (train.py / RayDPT) champions, filled in as they are found.
# ------------------------------------------------------------------
def _model_registry():
    import run_base
    import train
    return [
        ('batvision',           run_base, os.path.join(ROOT, 'checkpoints', 'batvision', 'best_model.pth')),
        ('best1 (my model)',    train,    os.path.join(ROOT, 'checkpoints', 'best1', 'best_model.pth')),
        ('best2 (my model)',    train,    os.path.join(ROOT, 'checkpoints', 'best2', 'best_model.pth')),
    ]


def _default_cfg():
    """Nested cfg with the default representation (5ch), for dataset + model build."""
    from types import SimpleNamespace
    import run_base
    args = SimpleNamespace(
        dataset_dir='/home/rvi-lab/workspace/sound-spaces/dataset_simplified',
        mode='test', batch_size=4, epochs=1, lr=3e-4, optimizer='AdamW',
        num_workers=0, checkpoint=None, flip_aug=True, experiment_name='batvision',
        eval_on='val', vis_every=0, use_log=True, feat_L=True, feat_R=True,
        feat_ILD=True, feat_cosIPD=True, feat_sinIPD=True, max_iters=0, max_val_batches=0)
    return run_base.make_config(args)


# ==================================================================
# Qualitative comparison grid
# ==================================================================
def build_qualitative(n_scenes=7, out_path=None):
    import torch
    from prepare import SoundSpacesDataset, load_gt_rgb

    out_path = out_path or os.path.join(DISPLAY_DIR, 'qualitative.png')
    os.makedirs(DISPLAY_DIR, exist_ok=True)
    cfg = _default_cfg()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SoundSpacesDataset(cfg, split='val')

    # pick the first sample of the first n_scenes distinct scenes
    picks, seen = [], set()
    for i, (scene, idx) in enumerate(ds.samples):
        if scene not in seen:
            seen.add(scene); picks.append(i)
        if len(picks) >= n_scenes:
            break

    items = [ds[i] for i in picks]
    specs = torch.stack([it['spec'] for it in items]).to(dev)
    gts = [it['depth'][0].numpy() * MAX_DEPTH for it in items]
    keys = [it['key'] for it in items]

    # predictions per registered model (None if checkpoint missing)
    models = _model_registry()
    preds = {}   # label -> list-over-scenes of pred array or None
    for label, module, ckpt in models:
        if not os.path.exists(ckpt):
            preds[label] = [None] * len(items)
            continue
        try:
            model = module.build_model(cfg).to(dev).eval()
            state = torch.load(ckpt, map_location=dev, weights_only=False)
            model.load_state_dict(state['state_dict'])
            with torch.no_grad():
                D = model(specs)['D'][:, 0].cpu().numpy() * MAX_DEPTH
            preds[label] = [D[k] for k in range(len(items))]
        except Exception as e:
            print(f'[report] {label}: load/predict failed ({e}); rendering pending', flush=True)
            preds[label] = [None] * len(items)

    col_titles = ['RGB', 'GT depth'] + [m[0] for m in models]
    ncol, nrow = len(col_titles), len(items)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), squeeze=False)

    def _placeholder(a, text):
        a.imshow(np.zeros((10, 20, 3)) + 0.9)
        a.text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
               color='0.4', transform=a.transAxes)

    for r, it in enumerate(items):
        scene = keys[r].split('/')[0]
        # RGB (absent in dataset_simplified -> N/A)
        rgb = load_gt_rgb(cfg.dataset.dataset_dir, scene, keys[r].split('/')[1],
                          cfg.dataset.depth_type, gts[r].shape[0], gts[r].shape[1])
        if rgb is not None:
            ax[r][0].imshow(rgb)
        else:
            _placeholder(ax[r][0], 'no RGB\n(simplified set)')
        ax[r][1].imshow(gts[r], vmin=0, vmax=MAX_DEPTH, cmap='turbo')
        for c, (label, _, _) in enumerate(models):
            p = preds[label][r]
            if p is None:
                _placeholder(ax[r][2 + c], f'{label}\n(pending)')
            else:
                ax[r][2 + c].imshow(p, vmin=0, vmax=MAX_DEPTH, cmap='turbo')
        ax[r][0].set_ylabel(scene, fontsize=9)

    for c, t in enumerate(col_titles):
        ax[0][c].set_title(t, fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle('Qualitative: GT vs predicted ERP radial depth (turbo, 0–%dm)' % int(MAX_DEPTH),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=95, bbox_inches='tight')
    plt.close(fig)
    print(f'[report] wrote {out_path}  ({nrow} scenes x {ncol} cols)')
    return out_path


# ==================================================================
# Results parsing + progress plot + README table
# ==================================================================
def _composite(abs_rel, rmse, d1):
    return rmse / 1.6 + (1.0 - d1) / 0.46 + 0.35 * abs_rel


def read_results():
    """Return list of dicts from out/results.tsv (data rows only)."""
    if not os.path.exists(RESULTS_TSV):
        return []
    rows = []
    with open(RESULTS_TSV) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if not lines:
        return []
    header = lines[0].split('\t')
    for ln in lines[1:]:
        parts = ln.split('\t')
        if len(parts) < len(header):
            parts += [''] * (len(header) - len(parts))
        row = dict(zip(header, parts))
        rows.append(row)
    return rows


def build_progress(out_path=None):
    out_path = out_path or os.path.join(DISPLAY_DIR, 'score_progress.png')
    os.makedirs(DISPLAY_DIR, exist_ok=True)
    rows = read_results()

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float('nan')

    pts = []
    for r in rows:
        if r.get('status', '') == 'crash':
            continue
        ar, rm, d1 = _f(r.get('abs_rel')), _f(r.get('rmse')), _f(r.get('d1'))
        if not (np.isfinite(ar) and np.isfinite(rm) and np.isfinite(d1)) or rm == 0:
            continue
        pts.append((r.get('commit', '')[:7], ar, rm, d1, _composite(ar, rm, d1)))

    # one full-width graph per metric, stacked vertically (no side-by-side split).
    # (idx into pts tuple, colour, higher_is_better)
    metrics = [('RMSE', 2, '#ccbb44', False),
               ('ABS_REL', 1, '#228833', False),
               ('a1 (d1, δ<1.25)', 3, '#aa3377', True)]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.1 * len(metrics)), squeeze=False)
    axes = axes[:, 0]
    if not pts:
        for a in axes:
            a.text(0.5, 0.5, 'No experiments logged yet\n(run_base.py / train.py -> out/results.tsv)',
                   ha='center', va='center', fontsize=11, color='0.4', transform=a.transAxes)
            a.set_xticks([]); a.set_yticks([])
        axes[0].set_title('Performance progress — RMSE / ABS_REL / a1 vs experiment', fontsize=12)
    else:
        xs = list(range(1, len(pts) + 1))
        for a, (name, j, colour, hib) in zip(axes, metrics):
            ys = [p[j] for p in pts]
            run_best = (np.maximum if hib else np.minimum).accumulate(ys)
            bi = int(np.argmax(ys) if hib else np.argmin(ys))
            a.plot(xs, ys, 'o-', color=colour, label=name)
            a.plot(xs, run_best, '--', color='#ee6677',
                   label='running best (%s)' % ('max' if hib else 'min'))
            a.scatter([xs[bi]], [ys[bi]], s=90, color='#ee6677', zorder=5)
            a.set_ylabel(name); a.grid(alpha=0.3); a.legend(fontsize=8, loc='best')
        axes[-1].set_xlabel('experiment #')
        axes[0].set_title('Performance progress (%d runs) — higher a1 / lower RMSE,ABS_REL = better'
                          % len(pts), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=95, bbox_inches='tight')
    plt.close(fig)
    print(f'[report] wrote {out_path}  ({len(pts)} runs)')
    return out_path


def update_readme_table():
    """Rewrite the <!-- RESULTS:START/END --> block in README with a metrics table."""
    rows = read_results()
    lines = ['<!-- RESULTS:START -->',
             '| # | commit | ABS_REL | RMSE | d1 | composite | status | description |',
             '|---|---|---|---|---|---|---|---|']
    if not rows:
        lines.append('| — | — | — | — | — | — | — | *(no experiments logged yet)* |')
    else:
        def _f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float('nan')
        for i, r in enumerate(rows, 1):
            ar, rm, d1 = _f(r.get('abs_rel')), _f(r.get('rmse')), _f(r.get('d1'))
            comp = '%.4f' % _composite(ar, rm, d1) if (np.isfinite(ar) and np.isfinite(rm)
                                                       and np.isfinite(d1) and rm) else '—'
            lines.append('| %d | `%s` | %s | %s | %s | %s | %s | %s |' % (
                i, (r.get('commit', '') or '')[:7], r.get('abs_rel', '—'), r.get('rmse', '—'),
                r.get('d1', '—'), comp, r.get('status', ''), r.get('description', '')))
    lines.append('<!-- RESULTS:END -->')
    block = '\n'.join(lines)

    with open(README) as f:
        text = f.read()
    pat = re.compile(r'<!-- RESULTS:START -->.*?<!-- RESULTS:END -->', re.DOTALL)
    if pat.search(text):
        text = pat.sub(block, text)
    else:
        text = text.rstrip() + '\n\n## Results\n\n' + block + '\n'
    with open(README, 'w') as f:
        f.write(text)
    print(f'[report] updated README results table ({len(rows)} rows)')


# ==================================================================
# Image retention (prune per-epoch visualization dumps)
# ==================================================================
def prune_visualizations(keep_latest=6, dry_run=False):
    """Retention for outputs/**/visualizations/ep###_##.png dumps.

    KEEP: the earliest epoch present (initial), each epoch that was a
    validation-composite milestone is not tracked here so we approximate
    "meaningful progress" by keeping evenly-spaced epochs, plus the latest
    `keep_latest` epochs. DELETE everything else. Curated figures in
    out/display/ are never touched.
    """
    removed = 0
    for vis_dir in glob.glob(os.path.join(ROOT, 'outputs', '*', 'visualizations')):
        files = glob.glob(os.path.join(vis_dir, 'ep*_*.png'))
        epochs = sorted({int(m.group(1)) for f in files
                         if (m := re.search(r'ep(\d+)_', os.path.basename(f)))})
        if len(epochs) <= keep_latest + 2:
            continue
        keep = set()
        keep.add(epochs[0])                                   # initial
        keep.update(epochs[-keep_latest:])                    # latest good
        mids = epochs[1:-keep_latest]
        if mids:                                              # a few evenly-spaced milestones
            for j in np.linspace(0, len(mids) - 1, min(3, len(mids)), dtype=int):
                keep.add(mids[int(j)])
        for f in files:
            m = re.search(r'ep(\d+)_', os.path.basename(f))
            if m and int(m.group(1)) not in keep:
                removed += 1
                if not dry_run:
                    os.remove(f)
        print(f'[prune] {vis_dir}: kept epochs {sorted(keep)} of {len(epochs)}')
    print(f'[prune] {"would remove" if dry_run else "removed"} {removed} files')
    return removed


def main():
    p = argparse.ArgumentParser(description='Reporting / visualization')
    p.add_argument('cmd', choices=['qualitative', 'progress', 'readme', 'prune', 'all'])
    p.add_argument('--n-scenes', type=int, default=7)
    p.add_argument('--keep-latest', type=int, default=6)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prune', action='store_true', help='also prune in `all`')
    a = p.parse_args()

    if a.cmd == 'qualitative':
        build_qualitative(a.n_scenes)
    elif a.cmd == 'progress':
        build_progress()
    elif a.cmd == 'readme':
        update_readme_table()
    elif a.cmd == 'prune':
        prune_visualizations(a.keep_latest, a.dry_run)
    elif a.cmd == 'all':
        build_qualitative(a.n_scenes)
        build_progress()
        update_readme_table()
        if a.prune:
            prune_visualizations(a.keep_latest, a.dry_run)


if __name__ == '__main__':
    main()
