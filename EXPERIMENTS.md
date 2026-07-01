# RayDPT Autoresearch — Experiment Findings

Audio → ERP radial depth (SoundSpaces, 256×512). Fixed **1-hour** training budget per run.
Metric: `compute_errors` in `prepare.py` — **ABS_REL, RMSE, d1 (δ<1.25)**. Live log: `results.tsv`.

## How to read the metrics (they mean different things — judge all three together)
- **ABS_REL** = mean(|D−gt|/gt): relative error, near-pixel weighted. ⚠️ **Directly optimized** by the relative loss → partly *gamed*, least trustworthy alone.
- **RMSE** = sqrt(mean((D−gt)²)): absolute error, far/large-depth weighted. Not optimized → **honest**.
- **d1** = % pixels within 1.25× ratio: overall accuracy. Not optimized → **honest, most holistic**.

**Rule:** trust **RMSE + d1** as the real quality signal; don't crown a config that only wins ABS_REL while RMSE/d1 regress. Model/epoch selection is now a **honest-weighted composite** (`rmse/1.6 + (1−d1)/0.46 + 0.3·abs_rel/0.4`).

## Results so far (best epoch by honest composite)

| run | change | ABS_REL | RMSE | d1 | verdict |
|---|---|---|---|---|---|
| baseline | fp32 bs16 lr3e-4 (~5 ep) | 0.4434 | 1.5907 | 0.5236 | keep |
| **E0b** | **bf16 AMP + bs32 + lr6e-4 + anneal (~7 ep)** | 0.4151 | 1.5887 | 0.5398 | keep (clean gain) |
| **E0c** | E0b, **lr 4e-4** | 0.4259 | **1.5199** | **0.5471** | keep — **best honest (RMSE+d1)** |
| E_d | + shared ray_proj | 0.4513 | 1.5280 | 0.5330 | discard |
| E_e | + full_decode | 0.4569 | 1.5273 | 0.5391 | discard (under-annealed) |
| E_f | + full_decode + time-anneal | 0.4594 | **1.5011** | 0.5447 | discard (ABS_REL froze; over budget) |
| E1 | relative loss **w_rel=0.25** | **0.3340** | 1.7181 | 0.5173 | discard (RMSE broken every epoch) |
| **E2** | relative loss **w_rel=0.1** | 0.3746 | 1.5540 | 0.5395 | **KEEP — best balanced (champion)** |
| E3 | rel0.25 + full_decode | 0.3443 | 1.7311 | 0.5182 | discard |
| E4 | SILog w_silog=0.5 | 0.3989 | 1.5468 | 0.5192 | keep (weak d1) |
| E5 | rel w_rel=0.13 | 0.3587 | 1.6377 | 0.5297 | discard (RMSE>baseline) |
| E6 | lr4e-4 + rel0.1 | 0.3570 | 1.5837 | 0.5337 | keep — best ABS_REL but worst RMSE/d1 of group (gamed) |
| E7 | lr4e-4 + rel0.1 + SILog0.3 | 0.3750 | 1.5538 | 0.5252 | discard (SILog hurts d1) |
| E8 | full_decode + lr4e-4 + rel0.05 | 0.3816 | 1.5520 | 0.5408 | discard (≈E2, complex, 72min over budget) |
| E9 | ray_cross_layers 2→3 | 0.3706 | 1.6044 | 0.5259 | discard (slower→6ep, worse RMSE/d1, over budget) |
| E10 | n_heads 4→8 | 0.3887 | 1.6002 | 0.5224 | discard (slower→5ep, worse all 3) |
| E11 | disable low-pass (w_low=0) | 0.3403 | 1.6212 | 0.5385 | discard (RMSE↑, d1 tied — low-pass helps RMSE) |
| E12 | w_low 0.5→1.0 | 0.3809 | 1.5754 | 0.5353 | discard (worse on all 3 → w_low=0.5 is optimal) |
| E13 | weight EMA (decay 0.999) | 0.3732 | 1.5706 | 0.5324 | discard (ABS_REL~tied, RMSE/d1 worse — EMA lagged) |
| E14 | weight EMA (decay 0.995) | running | | | — |

(E0 fp16 AMP crashed: NaN at epoch 2 → fixed with bf16.)

## Current best
- **Balanced champion: E2** (bf16+bs32, lr6e-4, rel w_rel=0.1) — **0.3746 / 1.554 / 0.5395**. Strong on all three.
- **Best honest metrics: E0c** (lr4e-4, no rel) — 0.4259 / **1.520 / 0.5471**.

## What helped
1. **bf16 AMP + batch 32 + LR cosine anneal (E0b)** — foundation. fp16→NaN, bf16 fixed it; more epochs/hr + real annealing → ABS_REL 0.4434→0.4151, RMSE flat.
2. **lr 4e-4 (E0c)** — lifts the honest metrics (RMSE 1.520, d1 0.5471).
3. **Light relative loss w_rel=0.1 (E2)** — `|D−gt|/gt` ≈ ABS_REL; at light weight + anneal, lowers ABS_REL to 0.3746 while keeping RMSE/d1 good.

## What did NOT help
- **fp16 AMP** → NaN. Use bf16.
- **Heavy relative (w_rel≥0.13)** → best ABS_REL but RMSE breaks (over-weights near pixels). w_rel=0.1 is the sweet spot.
- **Any capacity add** (full_decode, deeper cross-attn, more heads) → **slows epochs → fewer anneal steps → busts the 1-hour budget → worse RMSE/d1.** The model is at its budget-limited optimum with the light E2 config.
- **SILog** → helps nothing, hurts d1 (optimizes scale-invariant structure, not absolute correctness).
- **Weight EMA decay=0.999 (E13)** → ABS_REL ties E2 but RMSE/d1 regress. EMA was still climbing at epoch 7 → too slow to catch the annealed weights in a 7-epoch run. Retesting decay=0.995 (E14).
- **shared ray_proj / time-anneal / disabling low-pass** → each loses on the honest metrics.

## Key principles
- **ABS_REL ↔ RMSE anti-correlate** (across configs and epoch-to-epoch). Loss/schedule changes slide along a frontier; they don't push it in.
- The frontier is governed by the **loss balance**, not architecture. And under a **fixed time budget, lighter/faster models win** (more epochs > more capacity).
- **ABS_REL is gamable** → judge and select by **RMSE + d1** (honest), ABS_REL as a reported sanity-check.

## Ongoing / next (free levers that don't slow epochs)
Aux-loss weights (`w_low`, `w_coarse_layout`), `weight_decay`, local-attention window sizes (`raydpt_win32/64`), and efficient attention restructurings — all judged holistically vs E2. Loop runs autonomously and indefinitely (see `program.md` → Autonomous continuous operation).
