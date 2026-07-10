#!/usr/bin/env python3
"""Record one finished run: parse its log -> results.tsv -> regenerate every figure.

Exists because doing this by hand loses steps. The qualitative figure in particular was
silently stale for three experiments: it needs a model forward pass, so it is easy to skip
"just this once" while the GPU is busy. This script never skips it -- it renders on CPU by
default (CUDA_VISIBLE_DEVICES="") so it can run while a scored experiment trains, without
touching the GPU or perturbing that run's wall-clock budget.

Usage:
    python utils/record_run.py --exp-id E7 --name batvision_5ch_log_noaux \
        --commit $(git rev-parse --short HEAD) --status keep \
        --desc "I6/I7 discriminator: aux losses zeroed"
    python utils/record_run.py --name <name> --dry-run      # parse only, change nothing
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_log(path):
    """Pull the final best-checkpoint metrics and the run's shape out of a training log."""
    txt = open(path).read()
    screening = 'SCREENING RUN' in txt
    best = re.findall(r'Best model saved \(score ([\d.]+) \| ABS_REL ([\d.]+) '
                      r'RMSE ([\d.]+) d1 ([\d.]+)\)', txt)
    if not best:
        raise SystemExit(f'{path}: no "Best model saved" line -- crashed? '
                         f'record status=crash manually with 0.0000 metrics.')
    comp, abs_rel, rmse, d1 = (float(x) for x in best[-1])
    epochs = len(re.findall(r'^Epoch \[', txt, re.M))
    vram = re.search(r'peak_vram_mb: ([\d.]+)', txt)
    secs = re.search(r'training_seconds: ([\d.]+)', txt)
    ep_times = [float(x) for x in re.findall(r'Time: ([\d.]+)s', txt)]
    # best epoch = index of the last "Best model saved" among validation blocks
    n_best = len(best)
    val_blocks = len(re.findall(r'Val Loss:', txt))
    return dict(screening=screening, composite=comp, abs_rel=abs_rel, rmse=rmse, d1=d1, epochs_ran=epochs,
                vram_gb=round(float(vram.group(1)) / 1000, 1) if vram else 0.0,
                sec_per_epoch=round(sum(ep_times) / len(ep_times), 1) if ep_times else 0.0,
                training_seconds=float(secs.group(1)) if secs else 0.0,
                n_best_updates=n_best, val_blocks=val_blocks)


def main():
    p = argparse.ArgumentParser(description='Record a finished run and refresh all figures')
    p.add_argument('--name', required=True, help='experiment name (out/logs/<name>.log)')
    p.add_argument('--exp-id', default='?')
    p.add_argument('--commit', default=None, help='commit that produced the run (default: HEAD)')
    p.add_argument('--status', default='keep', choices=['keep', 'discard', 'crash'])
    p.add_argument('--desc', default='')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--gpu', action='store_true',
                   help='render figures on the GPU (default: CPU, so a training run is undisturbed)')
    a = p.parse_args()

    log = os.path.join(ROOT, 'out', 'logs', f'{a.name}.log')
    m = parse_log(log)
    commit = a.commit or subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip()
    desc = a.desc or a.name
    row = (f"{commit}\t{m['abs_rel']:.4f}\t{m['rmse']:.4f}\t{m['d1']:.4f}\t"
           f"{m['vram_gb']}\t{a.status}\t{a.exp_id} {desc}\n")

    print(f"[{a.exp_id}] {a.name}")
    print(f"  composite {m['composite']:.4f}  abs_rel {m['abs_rel']:.4f}  "
          f"rmse {m['rmse']:.4f}  d1 {m['d1']:.4f}")
    print(f"  epochs_ran {m['epochs_ran']}  sec/epoch {m['sec_per_epoch']}  "
          f"peak {m['vram_gb']}GB  best-updates {m['n_best_updates']}/{m['val_blocks']} vals")
    if m['n_best_updates'] == m['val_blocks'] and m['val_blocks'] > 0:
        print("  !! best checkpoint is the LAST epoch -- the run had not converged (see D5)")
    if m['screening'] and a.status == 'keep':
        raise SystemExit(
            "  !! this is a SCREENING run (shortened time budget). epochs-fit is part of the score\n"
            "     (D5), so it is NOT comparable with a scored run and must not be filed as `keep`.\n"
            "     Re-run at the full budget, or record it with --status discard.")
    if a.dry_run:
        print('\n[dry-run] results.tsv row would be:\n  ' + row.rstrip())
        return

    with open(os.path.join(ROOT, 'out', 'results.tsv'), 'a') as f:
        f.write(row)
    print('  -> out/results.tsv')

    env = dict(os.environ)
    if not a.gpu:
        env['CUDA_VISIBLE_DEVICES'] = ''      # never steal the GPU from a running experiment
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'utils', 'report.py'), 'all'],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith('[report]'):
            print('  ' + ln)
    if r.returncode:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit('report.py all FAILED -- figures are stale, fix before committing')


if __name__ == '__main__':
    main()
