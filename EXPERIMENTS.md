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
| **E14** | **weight EMA (decay 0.995)** | **0.3606** | **1.5548** | **0.5438** | **KEEP — NEW CHAMPION (comp 2.234 < E2 2.253)** |
| E15 | E14 + peak LR 6e-4→8e-4 | 0.3622 | 1.5601 | 0.5336 | discard (worse on all 3; d1↓ — LR too high) |
| **E16** | **E14 + peak LR 6e-4→4e-4** | **0.3528** | **1.5504** | **0.5488** | **KEEP — NEW CHAMPION (comp 2.214, beats E14 on all 3)** |
| E17 | E16 + peak LR 4e-4→3e-4 | 0.3577 | 1.5559 | 0.5456 | discard (worse on all 3 — LR U-turns; 4e-4 is the floor) |
| E18 | E16 + w_coarse_layout 1.0→0.5 | 0.3554 | 1.5552 | 0.5447 | discard (worse on all 3 — layout reg at 1.0 is right) |
| E19 | E16 + EMA decay 0.995→0.99 | 0.3570 | 1.5532 | 0.5470 | discard (worse on all 3; EMA 0.995 is the sweet spot) |
| E20 | LR anneal→0 over 7ep | 0.3582 | **1.5482** | 0.5477 | discard (best RMSE ever, but ABS_REL/d1 worse — frontier trade) |
| E21 | E16 + weight_decay 1e-4→2e-4 | 0.3572 | 1.5520 | 0.5464 | discard (worse on all 3 — wd 1e-4 optimal) |
| **E22** | **coarse 16×32 ray↔ray self-attn** | **0.3578** | **1.5414** | **0.5506** | **KEEP — NEW CHAMPION (comp 2.209; best-ever RMSE & d1)** |
| E23 | E22 + 2nd coarse self-attn block (deeper) | 0.3577 | 1.5437 | 0.5479 | discard (RMSE/d1 worse — 512-token grid saturates at 1 block) |
| E24 | E22 + global self-attn at 32×64 | crash | | | discard (587s/ep busts budget, 31.8GB — 2048 tok too costly) |
| E25 | E22 coarse self-attn, heads 4→8 | 0.3574 | 1.5406 | 0.5488 | discard (d1 worse → composite loses; 4 heads simpler) |
| E26 | E22 + pooled 32×64 global attn (coarse cost) | 0.3569 | 1.5468 | 0.5484 | discard (RMSE/d1 worse — mid-scale global attn doesn't help) |
| **E27** | **coarse self-attn + angular-dist bias (geometry-aware)** | **0.3581** | **1.5354** | **0.5528** | **KEEP — NEW CHAMPION (comp 2.201; best-ever RMSE & d1)** |
| E28 | E27 + richer geom bias (add absolute ray elevation) | 0.3555 | 1.5455 | 0.5474 | discard (RMSE/d1 worse — elevation biases toward gamed ABS_REL) |
| **E29** | **E27 + gated DPT skips** | **0.3523** | **1.5307** | **0.5537** | **KEEP — NEW CHAMPION (comp 2.191; best-ever all 3)** |
| E30 | E29 + light depth-head Refine (64×128) | 0.3483 | 1.5464 | 0.5567 | discard (composite ~tied, RMSE +0.016 from +20s/ep; simpler E29 wins) |
| E31 | E29 + deeper coarse cross-attn (cr16: 2→3 blocks) | 0.3507 | 1.5397 | 0.5552 | discard (ABS_REL/d1 better but RMSE +0.009 → composite ~tied-loses) |
| E32 | E29 + w_rel 0.1→0.08 (loss rebalance) | 0.3596 | 1.5295 | 0.5532 | discard (ABS_REL worse → composite loses; w_rel=0.1 optimal) |
| E33 | E29 + edge-aware gradient-matching loss (w_grad=0.1) | 0.3495 | 1.5437 | 0.5533 | discard (best ABS_REL but RMSE +0.013 → composite loses) |
| **E34** | **E29 + edge-aware gradient loss w_grad=0.05** | **0.3512** | **1.5313** | **0.5545** | **KEEP — NEW CHAMPION (comp 2.189)** |
| E35 | E34 + gradient loss w_grad=0.03 (bracket) | running | | | — |

(E0 fp16 AMP crashed: NaN at epoch 2 → fixed with bf16.)

## Current best
- **CHAMPION: E34** (E29 + edge-aware gradient-matching loss, w_grad=0.05) — **0.3512 / 1.5313 / 0.5545**, honest composite **2.189**. Light edge loss lifts ABS_REL & d1 at ~equal RMSE (w_grad=0.1 in E33 hurt RMSE).
- E29 (E27 + gated DPT skips) — 0.3523 / 1.5307 / 0.5537, comp 2.191. Best-ever RMSE.
- E27 (E22 + geometry-aware coarse self-attn, cos-ang-dist bias) — 0.3581 / 1.5354 / 0.5528, comp 2.201.
- E22 (E16 + coarse 16×32 ray↔ray self-attn) — 0.3578 / 1.5414 / 0.5506, comp 2.209.
- E16 (EMA 0.995 + lr 4e-4 + w_rel 0.1) — 0.3528 / 1.5504 / 0.5488, comp 2.214.
- **Architecture lesson:** ray↔ray global self-attn at the COARSE layout scale helps (E22); geometry-aware (cos-ang-dist bias, E27) helps more; gated encoder skips (E29) help more still. Capacity adds saturate (depth E23, heads E25); mid-scale/32×64 attn (E24/E26), richer geom w/ elevation (E28) don't help.
- E14 (EMA 0.995 + lr 6e-4) — 0.3606 / 1.5548 / 0.5438 (comp 2.234).
- **LR × EMA interaction:** with EMA, honest metrics improve as peak LR drops **8e-4→6e-4→4e-4**, then **U-turn at 3e-4 (E17, worse)** → **4e-4 is the sweet spot**. EMA does the noise-averaging, so low LR keeps ABS_REL good AND wins RMSE/d1. LR axis now fully mapped.

## What helped
1. **bf16 AMP + batch 32 + LR cosine anneal (E0b)** — foundation. fp16→NaN, bf16 fixed it; more epochs/hr + real annealing → ABS_REL 0.4434→0.4151, RMSE flat.
2. **lr 4e-4 (E0c)** — lifts the honest metrics (RMSE 1.520, d1 0.5471).
3. **Light relative loss w_rel=0.1 (E2)** — `|D−gt|/gt` ≈ ABS_REL; at light weight + anneal, lowers ABS_REL to 0.3746 while keeping RMSE/d1 good.
4. **Weight EMA decay=0.995 (E14)** — evaluate/checkpoint the temporal weight average, not the raw iterate. FREE (no epoch slowdown). Smooths late-training noise → better ABS_REL & d1 at equal RMSE. Decay must be fast enough (~200-step window) to track annealed weights in a 7-epoch run; 0.999 (E13) lagged and lost.

## What did NOT help
- **fp16 AMP** → NaN. Use bf16.
- **Heavy relative (w_rel≥0.13)** → best ABS_REL but RMSE breaks (over-weights near pixels). w_rel=0.1 is the sweet spot.
- **Any capacity add** (full_decode, deeper cross-attn, more heads) → **slows epochs → fewer anneal steps → busts the 1-hour budget → worse RMSE/d1.** The model is at its budget-limited optimum with the light E2 config.
- **SILog** → helps nothing, hurts d1 (optimizes scale-invariant structure, not absolute correctness).
- **Weight EMA decay=0.999 (E13)** → too slow: EMA still climbing at epoch 7, RMSE/d1 regress vs E2. Fix = faster decay 0.995 (E14, now champion). Lesson: EMA window must be << run length.
- **shared ray_proj / time-anneal / disabling low-pass** → each loses on the honest metrics.

## Key principles
- **ABS_REL ↔ RMSE anti-correlate** (across configs and epoch-to-epoch). Loss/schedule changes slide along a frontier; they don't push it in.
- The frontier is governed by the **loss balance**, not architecture. And under a **fixed time budget, lighter/faster models win** (more epochs > more capacity).
- **ABS_REL is gamable** → judge and select by **RMSE + d1** (honest), ABS_REL as a reported sanity-check.

## Ongoing / next (free levers that don't slow epochs)
Aux-loss weights (`w_low`, `w_coarse_layout`), `weight_decay`, local-attention window sizes (`raydpt_win32/64`), and efficient attention restructurings — all judged holistically vs E2. Loop runs autonomously and indefinitely (see `program.md` → Autonomous continuous operation).
