#!/usr/bin/env python3
"""Measure RayDPT throughput on the real GPU and pick the batch size.

TIME_BUDGET is wall-clock, so epochs-fit is silently part of the score (D5). E4 fitted only
5 epochs at 713 s/epoch while batvision fitted 25. This script measures, rather than guesses,
what s/epoch each batch size actually gives, and writes the chosen batch to out/raydpt_batch.txt.

It takes the SAME eval_lock as a scored run, so it can never overlap one: launch it whenever,
it will wait its turn.

    python utils/bench_raydpt.py --iters 8 --batches 16 32 48 64
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.evallock import eval_lock          # noqa: E402
import train                                   # noqa: E402

N_TRAIN = 28800            # training samples (informational; epoch time is extrapolated)
BUDGET = train.TIME_BUDGET


def bench(batch, iters, amp, device):
    """Time a real train step: forward + composite_loss + backward + step."""
    sys.argv = ['train.py', '--mode', 'train', '--batch-size', str(batch), '--amp', amp]
    cfg = train.make_config(train.parse_args())
    mcfg = train.build_model_cfg(cfg)
    model = train.build_model(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    amp_on = (amp == 'bf16')

    spec = torch.randn(batch, cfg.dataset.in_ch, 256, 512, device=device)
    gt = torch.rand(batch, 1, 256, 512, device=device)
    mask = (gt > 0.05).float()

    torch.cuda.reset_peak_memory_stats()
    for i in range(iters + 2):                        # 2 warm-up iters (cudnn autotune, alloc)
        if i == 2:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp_on):
            out = model(spec)
            loss, _ = train.composite_loss(out, gt, mask, mcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt, spec, gt, mask, out, loss
    torch.cuda.empty_cache()
    return dt, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=8)
    p.add_argument('--batches', type=int, nargs='+', default=[16, 32, 48, 64])
    p.add_argument('--amp', nargs='+', default=['off', 'bf16'])
    p.add_argument('--vram-cap-gb', type=float, default=44.0, help='leave headroom on the 49GB card')
    p.add_argument('--target-epochs', type=int, default=25)
    a = p.parse_args()

    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    with eval_lock('bench_raydpt'):
        print(f'{"amp":>5} {"batch":>6} {"s/iter":>8} {"peak GB":>8} {"s/epoch":>9} '
              f'{"epochs in 1h":>13}')
        print('-' * 56)
        best = None
        for amp in a.amp:
            for b in a.batches:
                try:
                    dt, peak = bench(b, a.iters, amp, device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f'{amp:>5} {b:>6} {"OOM":>8}')
                    continue
                sec_ep = dt * (N_TRAIN / b)
                eps = BUDGET / sec_ep
                flag = ''
                if peak > a.vram_cap_gb:
                    flag = '  (over VRAM cap)'
                elif best is None or sec_ep < best[0]:
                    best = (sec_ep, b, amp, peak, eps)
                    flag = '  <- best so far'
                print(f'{amp:>5} {b:>6} {dt:8.3f} {peak:8.2f} {sec_ep:9.1f} {eps:13.1f}{flag}')

    if best is None:
        raise SystemExit('no configuration fit the VRAM cap')
    sec_ep, b, amp, peak, eps = best
    print('-' * 56)
    print(f'\nchosen: batch={b} amp={amp}  ->  {sec_ep:.1f} s/epoch, ~{eps:.1f} epochs in the 1h budget')
    print(f'peak {peak:.2f} GB')
    if eps < a.target_epochs:
        print(f'!! still short of the {a.target_epochs}-epoch target '
              f'({eps:.1f}). More work needed on I8.')
    else:
        print(f'target of {a.target_epochs} epochs is reachable.')
    with open(os.path.join(ROOT, 'out', 'raydpt_batch.txt'), 'w') as f:
        f.write(f'{b} {amp} {sec_ep:.1f} {eps:.1f}\n')
    print('wrote out/raydpt_batch.txt')


if __name__ == '__main__':
    main()
