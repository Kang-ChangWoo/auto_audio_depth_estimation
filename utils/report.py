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
    # (label, module, checkpoint, feat_kwargs). Each model predicts on ITS OWN input
    # representation (channel count / log), so 2ch and 5ch models coexist in one figure.
    # The batvision reference is shown as exactly ONE column per channel count, always the
    # NON-LOG (raw magnitude) variant; the log variants are still trained and logged to
    # out/results.tsv, they just don't get a column here.
    only_lr = dict(feat_ILD=False, feat_cosIPD=False, feat_sinIPD=False)
    return [
        ('batvision (2ch, nolog)', run_base,
         os.path.join(ROOT, 'checkpoints', 'batvision_2ch_nolog', 'best_model.pth'),
         dict(use_log=False, **only_lr)),                    # [L, R]
        ('batvision (5ch, nolog)', run_base,
         os.path.join(ROOT, 'checkpoints', 'batvision_5ch_nolog', 'best_model.pth'),
         dict(use_log=False)),                               # [L, R, ILD, cosIPD, sinIPD]
        # "my model" = the RayDPT lineage champion. Its ARCHITECTURE flags must match the
        # checkpoint, or load_state_dict finds missing keys and the tile renders "pending".
        ('current (my model)', train,
         os.path.join(ROOT, 'checkpoints', 'raydpt_e9_d32L1_b64', 'best_model.pth'),
         dict(decode_scale=32, ray_cross_layers=1)),   # E9: 5ch log, 32x64 decode, 1 cross layer
    ]


def _args(**over):
    """CLI-args namespace for run_base/train make_config (defaults = 5ch, log on, historical STFT).

    Must carry every attribute make_config reads, including the editable acoustic
    representation (stft_*). A checkpoint trained with a non-default window must be
    rendered with that same window, so pass stft_* through `over` for such models.
    """
    from types import SimpleNamespace
    a = dict(dataset_dir='/home/rvi-lab/workspace/sound-spaces/dataset_simplified',
             mode='test', batch_size=4, epochs=1, lr=3e-4, optimizer='AdamW',
             num_workers=0, checkpoint=None, flip_aug=True, experiment_name='x',
             eval_on='val', vis_every=0, use_log=True, feat_L=True, feat_R=True,
             feat_ILD=True, feat_cosIPD=True, feat_sinIPD=True,
             stft_nfft=512, stft_hop=160, stft_win=400,
             w_coarse_layout=1.0, w_low=0.5, feat_interp='nearest', amp='off',
             raydpt_lite=False, decode_scale=64, ray_cross_layers=2, ffn_mult=4, cross_kv32='e3',
             max_iters=0, max_val_batches=0)
    a.update(over)
    return SimpleNamespace(**a)


# ==================================================================
# Qualitative comparison grid
# ==================================================================
def build_qualitative(n_scenes=7, out_path=None):
    import torch
    import run_base
    from prepare import SoundSpacesDataset, load_gt_rgb

    out_path = out_path or os.path.join(DISPLAY_DIR, 'qualitative.png')
    os.makedirs(DISPLAY_DIR, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # base dataset (any representation) for GT / keys / scene picks — all config-independent
    base_ds = SoundSpacesDataset(run_base.make_config(_args()), split='val')
    dataset_dir, depth_type = base_ds.root_dir, base_ds.depth_type

    # pick the first sample of the first n_scenes distinct scenes
    picks, seen = [], set()
    for i, (scene, idx) in enumerate(base_ds.samples):
        if scene not in seen:
            seen.add(scene); picks.append(i)
        if len(picks) >= n_scenes:
            break

    gts, keys = [], []
    for i in picks:
        it = base_ds[i]
        gts.append(it['depth'][0].numpy() * MAX_DEPTH); keys.append(it['key'])

    # each model predicts on ITS OWN input representation (build a matching dataset + cfg)
    models = _model_registry()
    preds = {}          # label -> list-over-scenes of pred array or None
    ds_cache = {}
    for label, module, ckpt, feat in models:
        if not os.path.exists(ckpt):
            preds[label] = [None] * len(picks)
            continue
        try:
            cfg_m = module.make_config(_args(**feat))
            fkey = tuple(sorted(feat.items()))
            if fkey not in ds_cache:
                ds_cache[fkey] = SoundSpacesDataset(cfg_m, split='val')
            ds_m = ds_cache[fkey]
            specs = torch.stack([ds_m[i]['spec'] for i in picks]).to(dev)
            model = module.build_model(cfg_m).to(dev).eval()
            state = torch.load(ckpt, map_location=dev, weights_only=False)
            # A checkpoint may carry EXTRA keys the current model no longer has (I9 deleted
            # RayDPT's dead e5..e8 tail). Extra keys are inert and are dropped loudly. MISSING
            # keys are not: they would silently render a partly-random model as if it were real.
            sd, own = state['state_dict'], model.state_dict()
            missing = [k for k in own if k not in sd]
            extra = [k for k in sd if k not in own]
            if missing:
                raise RuntimeError(f'checkpoint lacks {len(missing)} keys the model needs '
                                   f'(e.g. {missing[:2]}) -- refusing to render a partial model')
            if extra:
                print(f'[report] {label}: dropping {len(extra)} stale checkpoint keys '
                      f'(e.g. {extra[0]})', flush=True)
            model.load_state_dict({k: v for k, v in sd.items() if k in own})
            with torch.no_grad():
                D = model(specs)['D'][:, 0].cpu().numpy() * MAX_DEPTH
            preds[label] = [D[k] for k in range(len(picks))]
        except Exception as e:
            print(f'[report] {label}: load/predict failed ({e}); rendering pending', flush=True)
            preds[label] = [None] * len(picks)

    col_titles = ['RGB', 'GT depth'] + [m[0] for m in models]
    ncol, nrow = len(col_titles), len(picks)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), squeeze=False)

    def _placeholder(a, text):
        a.imshow(np.zeros((10, 20, 3)) + 0.9)
        a.text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
               color='0.4', transform=a.transAxes)

    for r in range(nrow):
        scene = keys[r].split('/')[0]
        # RGB (absent in dataset_simplified -> N/A)
        rgb = load_gt_rgb(dataset_dir, scene, keys[r].split('/')[1],
                          depth_type, gts[r].shape[0], gts[r].shape[1])
        if rgb is not None:
            ax[r][0].imshow(rgb)
        else:
            _placeholder(ax[r][0], 'no RGB\n(simplified set)')
        ax[r][1].imshow(gts[r], vmin=0, vmax=MAX_DEPTH, cmap='turbo')
        for c, entry in enumerate(models):
            label = entry[0]
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
    fig.suptitle('Qualitative: GT vs predicted ERP planar (cubemap) depth (turbo, 0–%dm)' % int(MAX_DEPTH),
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


def update_readme_research():
    """Rewrite the <!-- RESEARCH:START/END --> block: the live autonomous-research dashboard.

    Independent of the RESULTS block, which stays exactly as it is. A reader opening the
    README should see, without clicking anything: what mode the researcher is in, what it is
    asking, what it just learned, what it will do next, which ideas are alive, and which
    observations remain unexplained. This is a dashboard, NOT a database -- no full log dumps.
    """
    import json
    sys.path.insert(0, ROOT)
    from utils.research import (load_studies, load_ideas, recent_decisions, BUDGET)

    st = load_studies()
    ideas = load_ideas()
    a = st.get('active_study', {})
    mode = st.get('mode', 'exploit')
    runs = a.get('runs', [])
    latest = runs[-1] if runs else None

    L = ['<!-- RESEARCH:START -->',
         '## Autonomous research state', '',
         '| | |', '|---|---|',
         f'| **Mode** | `{mode.upper()}` — {BUDGET.get(mode, "")} |',
         f'| **Active study** | `{a.get("study_id","—")}` [{a.get("type","—")}] '
         f'{a.get("lineage","—")} (*{a.get("status","—")}*) |',
         f'| **Research question** | {a.get("general_hypothesis","—")[:200]} |',
         f'| **Current action** | {a.get("experiment_note","—")[:180]} |']
    if latest:
        L.append(f'| **Latest result** | `{latest.get("exp_id")}` {latest.get("name","")}: '
                 f'composite **{latest.get("composite")}** '
                 f'(rmse {latest.get("rmse")}, d1 {latest.get("d1")}, abs_rel {latest.get("abs_rel")}), '
                 f'best epoch {latest.get("best_epoch")}/{latest.get("epochs_ran")} |')
    else:
        L.append('| **Latest result** | *(no scored run in this study yet)* |')
    L.append(f'| **Next decision** | {a.get("decision_rule","—")[:200]} |')
    L.append(f'| **Why this mode** | {st.get("mode_reason","—")[:200]} |')
    L += ['', '### Current hypothesis', '',
          f'- **General** — {a.get("general_hypothesis","—")}',
          f'- **Detailed** — {a.get("detailed_hypothesis","—")}',
          f'- **Implementation note** — {a.get("experiment_note","—")}', '']

    live = [i for i in ideas.get('ideas', []) if i.get('status') not in ('dropped', 'validated')]
    if live:
        L += ['### Research portfolio', '',
              '| Idea | Mechanism family | Causal distance | Target bottleneck | Status | Next test |',
              '|---|---|---|---|---|---|']
        for i in live:
            L.append('| `%s` | %s | %s | %s | %s | %s |' % (
                i['id'], i.get('mechanism_family', '—'), i.get('causal_distance', '—'),
                i.get('target_bottleneck', '—'), i.get('status', '—'),
                i.get('next_action', '—')[:90]))
        L.append('')

    op = [d for d in ideas.get('discrepancies', []) if d.get('status') == 'open']
    if op:
        L += ['### Open discrepancies', '',
              '*Unexplained observations are research assets, not noise.*', '']
        for d in op:
            L.append(f'- **`{d["id"]}`** — {d.get("observation","")}')
            L.append(f'  <br/>*Why it matters:* {d.get("why_it_matters","")}')
        L.append('')

    dec = recent_decisions(8)
    if dec:
        L += ['### Recent decisions', '', '| When | Mode | Event | Note |', '|---|---|---|---|']
        for d in reversed(dec):
            note = (d.get('note') or d.get('reason') or '').replace('|', '\\|')
            L.append('| %s | `%s` | %s | %s |' % (
                d.get('ts', '')[:16], d.get('mode', '?'), d.get('event', '?'), note[:130]))
        L.append('')

    L.append(f'*Updated by `python utils/report.py research`. '
             f'Champion: {st.get("global_champion") or "none yet"}.*')
    L.append('<!-- RESEARCH:END -->')
    block = '\n'.join(L)

    with open(README) as f:
        text = f.read()
    pat = re.compile(r'<!-- RESEARCH:START -->.*?<!-- RESEARCH:END -->', re.DOTALL)
    if pat.search(text):
        text = pat.sub(block, text)
    else:
        # first install: insert near the top, right after the title + tagline paragraph
        parts = text.split('\n\n', 2)
        text = ('\n\n'.join(parts[:2]) + '\n\n' + block + '\n\n' + parts[2]) if len(parts) > 2 \
            else text.rstrip() + '\n\n' + block + '\n'
    with open(README, 'w') as f:
        f.write(text)
    print(f'[report] updated README research dashboard (mode={mode}, '
          f'{len(live)} live ideas, {len(op)} open discrepancies)')


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
    p.add_argument('cmd', choices=['qualitative', 'progress', 'readme', 'research', 'prune', 'all'])
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
    elif a.cmd == 'research':
        update_readme_research()
    elif a.cmd == 'prune':
        prune_visualizations(a.keep_latest, a.dry_run)
    elif a.cmd == 'all':
        build_qualitative(a.n_scenes)
        build_progress()
        update_readme_table()
        update_readme_research()
        if a.prune:
            prune_visualizations(a.keep_latest, a.dry_run)


if __name__ == '__main__':
    main()
