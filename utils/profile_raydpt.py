#!/usr/bin/env python3
"""Where does RayDPT's GPU time actually go, and which LSA implementation is fastest?

Written after guessing twice and being wrong twice. A CPU forward-time proxy said
LocalSphericalAttention was 45% of the cost; two "optimisations" derived from that proxy
both made things worse (an accumulate loop doubled VRAM; an as_strided einsum made bf16
1.5x SLOWER than fp32). Measure on the real device, with backward included, or don't claim.

Takes the eval_lock, so it never overlaps a scored run.

    python utils/profile_raydpt.py
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.evallock import eval_lock     # noqa: E402
import train                              # noqa: E402


def timed(fn, iters=10, warm=3):
    """Median wall time of fn() including backward, with CUDA synchronisation."""
    ts = []
    for i in range(iters + warm):
        torch.cuda.synchronize(); t0 = time.time()
        fn()
        torch.cuda.synchronize()
        if i >= warm:
            ts.append(time.time() - t0)
    ts.sort()
    return ts[len(ts) // 2]


# ---- the three LSA implementations, isolated -------------------------------------------
class LSA_unfold(nn.Module):
    """the original: F.unfold materialises (B, C*win*win, H*W)"""
    def __init__(self, dim, heads, H, W, win):
        super().__init__()
        self.h, self.dh, self.win = heads, dim // heads, win
        self.scale = self.dh ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1); self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        p, win = self.win // 2, self.win
        q, k, v = self.to_qkv(x).chunk(3, 1)
        def wkv(t):
            t = torch.cat([t[..., -p:], t, t[..., :p]], dim=-1)
            t = F.pad(t, (0, 0, p, p), mode="replicate")
            return F.unfold(t, kernel_size=win).view(B, C, win * win, H, W)
        kw = wkv(k).view(B, self.h, self.dh, win * win, H, W)
        vw = wkv(v).view(B, self.h, self.dh, win * win, H, W)
        q = q.view(B, self.h, self.dh, H, W)
        a = torch.einsum("bndhw,bndkhw->bnkhw", q, kw) * self.scale
        a = a.softmax(dim=2)
        o = torch.einsum("bnkhw,bndkhw->bndhw", a, vw).reshape(B, C, H, W)
        return x + self.proj(o)


class LSA_loop(nn.Module):
    """accumulate over offsets: no unfold, but win*win intermediates retained for autograd"""
    def __init__(self, dim, heads, H, W, win):
        super().__init__()
        self.h, self.dh, self.win = heads, dim // heads, win
        self.scale = self.dh ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1); self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        p, win = self.win // 2, self.win
        q, k, v = self.to_qkv(x).chunk(3, 1)
        q = q.view(B, self.h, self.dh, H, W)
        def pad(t):
            t = torch.cat([t[..., -p:], t, t[..., :p]], dim=-1)
            return F.pad(t, (0, 0, p, p), mode="replicate")
        kp, vp = pad(k), pad(v)
        logits = [(q * kp[..., dr:dr + H, dc:dc + W].view(B, self.h, self.dh, H, W)).sum(2)
                  for dr in range(win) for dc in range(win)]
        a = (torch.stack(logits, 2) * self.scale).softmax(dim=2)
        o = x.new_zeros(B, self.h, self.dh, H, W)
        for idx, (dr, dc) in enumerate((r, c) for r in range(win) for c in range(win)):
            o = o + a[:, :, idx].unsqueeze(2) * vp[..., dr:dr + H, dc:dc + W].view(B, self.h, self.dh, H, W)
        return x + self.proj(o.reshape(B, C, H, W))


class LSA_strided(nn.Module):
    """as_strided view + einsum: no copy, but einsum on a 7D non-contiguous view"""
    def __init__(self, dim, heads, H, W, win):
        super().__init__()
        self.h, self.dh, self.win = heads, dim // heads, win
        self.scale = self.dh ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1); self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        p, win = self.win // 2, self.win
        q, k, v = self.to_qkv(x).chunk(3, 1)
        q = q.view(B, self.h, self.dh, H, W)
        def neigh(t):
            t = torch.cat([t[..., -p:], t, t[..., :p]], dim=-1)
            t = F.pad(t, (0, 0, p, p), mode="replicate")
            Hp, Wp = t.shape[-2:]
            t = t.view(B, self.h, self.dh, Hp, Wp)
            sB, sh, sd, sH, sW = t.stride()
            return t.as_strided((B, self.h, self.dh, win, win, H, W), (sB, sh, sd, sH, sW, sH, sW))
        kw, vw = neigh(k), neigh(v)
        a = torch.einsum('bndhw,bndrshw->bnrshw', q, kw) * self.scale
        a = a.reshape(B, self.h, win * win, H, W).softmax(2).view(B, self.h, win, win, H, W)
        o = torch.einsum('bnrshw,bndrshw->bndhw', a, vw)
        return x + self.proj(o.reshape(B, C, H, W))


def main():
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    with eval_lock('profile_raydpt'):
        print('=== LSA implementations, fwd+bwd, batch 16 (ms, median) ===')
        print(f'{"impl":>12} {"32x64 win5":>13} {"64x128 win3":>13} {"peak GB":>9}  dtype')
        print('-' * 58)
        for amp in (False, True):
            for name, cls in [('unfold', LSA_unfold), ('loop', LSA_loop), ('strided', LSA_strided)]:
                row, peak = [], 0.0
                for (H, W, win) in [(32, 64, 5), (64, 128, 3)]:
                    m = cls(192, 4, H, W, win).to(dev)
                    x = torch.randn(16, 192, H, W, device=dev, requires_grad=True)
                    torch.cuda.reset_peak_memory_stats()
                    def step():
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                            y = m(x)
                        y.sum().backward()
                    row.append(timed(step) * 1e3)
                    peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
                    del m, x; torch.cuda.empty_cache()
                print(f'{name:>12} {row[0]:13.1f} {row[1]:13.1f} {peak:9.2f}  '
                      f'{"bf16" if amp else "fp32"}')
        print('-' * 58)

        # ---- whole-model module breakdown, using the CURRENT train.py ----
        print('\n=== RayDPT module breakdown (fwd only, batch 16, bf16) ===')
        sys.argv = ['train.py', '--mode', 'train', '--batch-size', '16', '--amp', 'bf16']
        cfg = train.make_config(train.parse_args())
        m = train.build_model(cfg).to(dev).train()
        x = torch.randn(16, cfg.dataset.in_ch, 256, 512, device=dev)
        acc = {}
        hooks = []
        def mk(name):
            def pre(mod, inp):
                torch.cuda.synchronize(); mod._t0 = time.time()
            def post(mod, inp, out):
                torch.cuda.synchronize(); acc[name] = acc.get(name, 0) + time.time() - mod._t0
            return pre, post
        for name in ['enc', 'cr16', 'cr32', 'cr64', 'lsa32', 'lsa64', 'head', 'refine32', 'refine64']:
            mod = getattr(m, name, None)
            if mod is None:
                continue
            pre, post = mk(name)
            hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(3):
                m(x)                       # warm
            acc.clear()
            for _ in range(5):
                m(x)
        for h in hooks:
            h.remove()
        tot = sum(acc.values())
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f'  {k:>10} {v/5*1e3:8.1f} ms   {v/tot*100:5.1f}%')
        print(f'  {"TOTAL":>10} {tot/5*1e3:8.1f} ms   (hooked modules only)')


if __name__ == '__main__':
    main()
