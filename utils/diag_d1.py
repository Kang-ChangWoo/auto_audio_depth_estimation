#!/usr/bin/env python3
"""WHERE does RayDPT lose d1 to the reference? Full val set, on GPU, under the eval_lock.

D9: with both models converged, RayDPT's whole deficit is d1 (angle), not rmse (range). This
splits that scalar into regions, which is what separates the surviving hypotheses:

  H1 ray-conditioning does not aid angular assignment    -> deficit spread evenly over azimuth
  H4 two receivers under-determine fine azimuth          -> deficit concentrated where binaural
                                                            cues are weakest: the front/back cones
                                                            (|y| small), where ILD and IPD vanish
  H3 raw capacity (5.89M vs 54.41M)                      -> deficit largest where the scene is
                                                            most complex (near surfaces, high GT
                                                            depth variance)

The front/back confusion cone is the sharp prediction: for a two-microphone array, ILD and IPD
are near-zero along the x axis (front and back), so azimuth there is genuinely ambiguous. If
RayDPT's deficit lives THERE, it is a sensing limit that ray conditioning cannot fix. If it is
flat across azimuth, ray conditioning is simply not paying for itself.

Per D10: full val set only, and the raw metrics are checked against the training logs first.

    python utils/diag_d1.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.evallock import eval_lock          # noqa: E402
from utils.report import _args                 # noqa: E402
from prepare import make_dataloader, erp_grid  # noqa: E402
import run_base                                # noqa: E402
import train                                   # noqa: E402

MAX = 10.0
MODELS = {
    'batvision E3': (run_base, 'checkpoints/batvision_5ch_log/best_model.pth',
                     dict(use_log=True)),
    'RayDPT E11': (train, 'checkpoints/raydpt_e11_d32L2_kve4/best_model.pth',
                   dict(use_log=True, decode_scale=32, ray_cross_layers=2, cross_kv32='e4')),
}
OFFICIAL = {'batvision E3': (0.4517, 1.3088, 0.5949),
            'RayDPT E11': (0.4199, 1.3276, 0.5710)}


def collect(module, ckpt, dev, **feat):
    cfg = module.make_config(_args(**feat))
    cfg.mode.batch_size, cfg.mode.num_threads = 32, 8
    _, loader = make_dataloader(cfg, 'val', shuffle=False)
    m = module.build_model(cfg).to(dev).eval()
    sd = torch.load(os.path.join(ROOT, ckpt), map_location=dev, weights_only=False)['state_dict']
    own = m.state_dict()
    miss = [k for k in own if k not in sd]
    assert not miss, f'checkpoint missing {miss[:3]}'
    m.load_state_dict({k: v for k, v in sd.items() if k in own})

    hit = torch.zeros(256, 512, device=dev)      # per-pixel count of d1 hits
    cnt = torch.zeros(256, 512, device=dev)
    dep_hit = torch.zeros(10, device=dev); dep_cnt = torch.zeros(10, device=dev)
    tot = [0.0, 0.0, 0.0]; n = 0
    with torch.no_grad():
        for b in loader:
            D = m(b['spec'].to(dev))['D'][:, 0].float() * MAX
            G = b['depth'][:, 0].to(dev) * MAX
            valid = G > 0
            P = D.clamp(min=1e-3)
            thresh = torch.maximum(G / P, P / G.clamp(min=1e-6))
            ok = (thresh < 1.25) & valid
            hit += ok.sum(0); cnt += valid.sum(0)
            # per-sample metrics, matching compute_errors' per-image mean
            for i in range(G.shape[0]):
                v = valid[i]
                if v.sum() < 100:
                    continue
                g, p = G[i][v], P[i][v]
                th = torch.maximum(g / p, p / g)
                tot[0] += ((g - p).abs() / g).mean().item()
                tot[1] += torch.sqrt(((g - p) ** 2).mean()).item()
                tot[2] += (th < 1.25).float().mean().item()
                n += 1
            # d1 by GT depth decile
            bins = (G / MAX * 10).long().clamp(0, 9)
            for k in range(10):
                sel = valid & (bins == k)
                dep_hit[k] += ok[sel].sum(); dep_cnt[k] += sel.sum()
    return (np.array(tot) / n, (hit / cnt.clamp(min=1)).cpu().numpy(),
            (dep_hit / dep_cnt.clamp(min=1)).cpu().numpy())


def main():
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    with eval_lock('diag_d1'):
        res = {}
        for nm, (mod, ck, ft) in MODELS.items():
            res[nm] = collect(mod, ck, dev, **ft)
            e = res[nm][0]; o = OFFICIAL[nm]
            ok = abs(e[1] - o[1]) < 5e-3 and abs(e[2] - o[2]) < 5e-3
            print(f'[integrity] {nm:14s} abs_rel {e[0]:.4f} rmse {e[1]:.4f} d1 {e[2]:.4f}  '
                  f'(log: {o[0]:.4f} {o[1]:.4f} {o[2]:.4f})  {"OK" if ok else "MISMATCH"}')
            assert ok, 'instrument does not reproduce the training log -- fix before concluding'

    el, az = erp_grid(256, 512)
    d1b, d1r = res['batvision E3'][1], res['RayDPT E11'][1]
    diff = d1b - d1r                      # positive = batvision better here

    print('\n=== d1 deficit by AZIMUTH sector (positive = RayDPT worse) ===')
    print('    ILD/IPD vanish near az = 0 (front) and az = +-180 (back): the confusion cone.')
    print(f'{"sector":>22} {"batvision d1":>13} {"RayDPT d1":>11} {"deficit":>9}')
    print('-' * 60)
    a = np.abs(np.degrees(az))
    sectors = [('front  |az|<30', a < 30), ('front-side 30-60', (a >= 30) & (a < 60)),
               ('side   60-120', (a >= 60) & (a < 120)), ('back-side 120-150', (a >= 120) & (a < 150)),
               ('back   |az|>150', a >= 150)]
    for nm, mk in sectors:
        print(f'{nm:>22} {d1b[mk].mean():13.4f} {d1r[mk].mean():11.4f} {diff[mk].mean():9.4f}')

    print('\n=== d1 deficit by ELEVATION ===')
    e = np.degrees(el)
    for nm, mk in [('up   el>30', e > 30), ('mid  |el|<30', np.abs(e) <= 30), ('down el<-30', e < -30)]:
        print(f'{nm:>22} {d1b[mk].mean():13.4f} {d1r[mk].mean():11.4f} {diff[mk].mean():9.4f}')

    print('\n=== d1 by GT depth decile (0-1m ... 9-10m) ===')
    db, dr = res['batvision E3'][2], res['RayDPT E11'][2]
    print(f'{"depth":>10} {"batvision":>11} {"RayDPT":>9} {"deficit":>9}')
    for k in range(10):
        print(f'{f"{k}-{k+1}m":>10} {db[k]:11.4f} {dr[k]:9.4f} {db[k]-dr[k]:9.4f}')

    print(f'\noverall d1 deficit: {diff.mean():.4f}')
    print(f'deficit std across azimuth sectors: '
          f'{np.std([diff[mk].mean() for _, mk in sectors]):.4f}  '
          f'(flat => H1; concentrated in front/back => H4)')


if __name__ == '__main__':
    main()
