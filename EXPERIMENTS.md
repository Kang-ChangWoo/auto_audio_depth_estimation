# RayDPT Autoresearch — Experiment Findings

Audio → ERP radial depth (SoundSpaces, 256×512). Fixed **1-hour** training budget per run.
Metric: `compute_errors` in `prepare.py` — **ABS_REL, RMSE, d1 (δ<1.25)**. Live log: `results.tsv`.

## How to read the metrics (they mean different things — judge all three together)
- **ABS_REL** = mean(|D−gt|/gt): relative error, near-pixel weighted. ⚠️ **Directly optimized** by the relative loss → partly *gamed*, least trustworthy alone.
- **RMSE** = sqrt(mean((D−gt)²)): absolute error, far/large-depth weighted. Not optimized → **honest**.
- **d1** = % pixels within 1.25× ratio: overall accuracy. Not optimized → **honest, most holistic**.

**Rule:** trust **RMSE + d1** as the real quality signal; don't crown a config that only wins ABS_REL while RMSE/d1 regress. Model/epoch selection is now a **honest-weighted composite** (`rmse/1.6 + (1−d1)/0.46 + 0.3·abs_rel/0.4`).

**Noise floor (E36 rerun of E34):** identical config reruns differ by **~0.0045 composite** (~0.008 RMSE). So only improvements **> ~0.005 composite** are real. The big architectural steps (E22/E27/E29, Δ≈0.008–0.01) are real; sub-0.005 loss-weight "wins" (E34 vs E29 = 0.002) are within noise. E34 ≈ E29.

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
| E35 | E34 + gradient loss w_grad=0.03 (bracket) | 0.3525 | 1.5448 | 0.5548 | discard (RMSE +0.014; non-monotonic sweep ⇒ ~0.01 RMSE noise) |
| E36 | Confirmation rerun of E34 (noise gauge) | 0.3515 | 1.5389 | 0.5547 | keep (rerun; comp 2.194 vs E34 2.189 → **noise floor ~0.0045**) |
| E37 | E34 + 2nd coarse read of hi-res audio (kv_e3) | 0.3508 | 1.5531 | 0.5473 | discard (BUDGET BUST: 574s/ep → only 6 epochs, under-annealed, comp 2.218) |
| E38 | E34 + berHu main loss (was MAE) | 0.3775 | **1.4746** | 0.5424 | discard (RMSE massive best-ever −0.057 but ABS_REL/d1 sink composite) |
| E39 | E34 + 0.5·MAE + 0.5·berHu blend | 0.3633 | 1.4999 | 0.5496 | discard (=tie E34; MAE↔berHu is a flat frontier slide) |
| E40 | E39 blend + w_rel 0.1→0.13 (combine berHu+rel) | 0.3588 | 1.5240 | 0.5467 | discard (loses 0.018; combo compounds d1 damage — berHu exhausted) |
| E41 | E34 + lsa64 window 3→5 | crash | | | discard (709s/ep — win25 einsum too costly; killed ep1) |
| E42 | E34 + batch size 32→40 | 0.3507 | 1.5352 | 0.5552 | discard (=tie E34, within noise; batch size neutral) |
| E43 | E34 + coarse-to-fine guidance (inject d_c into decoder) | 0.3542 | 1.5417 | 0.5551 | discard (loses 0.007; layout already in decoder feats) |
| E44 | E34 + global-audio FiLM conditioning of decoder | 0.3552 | 1.5401 | 0.5542 | discard (loses 0.009; audio cond. already saturated by cross-attn) |
| E45 | E34 + SwiGLU FFN in coarse GeoSelfBlock | 0.3544 | 1.5388 | 0.5520 | discard (loses 0.012; coarse block saturated) |
| E46 | E34 + log-depth L1 aux loss (w_logd=0.1) | 0.3614 | 1.5234 | 0.5550 | discard (=tie, within noise; another RMSE↔ABS_REL frontier slide) |
| E47 | E34 + learnable attention temperature (coarse block) | 0.3502 | 1.5368 | 0.5540 | discard (=tie, within noise; fixed scale fine) |
| E48 | E34 + EMA warmup (skip averaging noisy 1st epoch) | 0.3515 | 1.5415 | 0.5522 | discard (loses 0.012; less averaging hurts — constant EMA better) |
| E49 | E34 + EMA decay 0.995→0.997 (bracket fill) | 0.3513 | 1.5364 | 0.5512 | discard (loses 0.010; 0.997 lags → 0.995 optimal, axis mapped) |
| **E50** | **2nd geo self-attn on FUSED coarse m16** | **0.3455** | **1.5204** | **0.5619** | **KEEP — NEW CHAMPION (comp 2.162, beats E34 by 0.027 ≫ noise; best-ever all 3)** |
| **E51** | **post-fusion geo-attn 2 blocks** | **0.3390** | **1.5105** | **0.5637** | **KEEP — NEW CHAMPION (comp 2.147, beats E50 by 0.015; best-ever all 3)** |
| E52 | E51 + post-fusion geo-attn 3 blocks | 0.3433 | 1.5156 | 0.5618 | discard (saturates; loses 0.011) |
| E53 | E51 + geo-aware pooled geo-attn at 32×64 (breadth) | crash | | | discard (556s/ep > budget ceiling; killed — budget now binding) |
| **E54** | **E51 − pre-fusion geo-attn (rsa16)** | **0.3426** | **1.5085** | **0.5682** | **KEEP — NEW CHAMPION (comp 2.139, beats E51 by 0.008; best-ever RMSE & d1; simpler+faster)** |
| E55 | E54 + geo-aware pooled geo-attn at 32×64 | 0.3418 | 1.5084 | 0.5662 | discard (=tie E54, within noise; pooled adds nothing over lsa32) |
| **E56** | **E54 + richer geom (cos-dist+elev) on post-fusion rsa16b** | **0.3410** | **1.4950** | **0.5745** | **KEEP — NEW CHAMPION (comp 2.115, beats E54 by 0.023; best-ever all 3, RMSE<1.50)** |
| **E57** | **E56 + wrapped Δazimuth geom (5 feats)** | **0.3393** | **1.4920** | **0.5767** | **KEEP — NEW CHAMPION (comp 2.107, beats E56 by 0.008; best-ever all 3)** |
| E58 | E57 + wider geom bias_mlp (32→64) | 0.3437 | 1.5023 | 0.5725 | discard (loses 0.019; capacity overfits — geom features help, not bias_mlp width) |
| E59 | E57 + 3rd post-fusion rsa16b block | — | — | — | discard (BUDGET BUST 560s>555s → 6 epochs; 2 blocks is the sweet spot) |
| E60 | E57 confirmation rerun | 0.3417 | 1.5074 | 0.5697 | confirm — CRITICAL: identical config 0.027 WORSE → true σ≈0.019, recent micro-wins were noise |
| E61 | E57 + 2nd cross-attn round at m16 | 0.3382 | 1.4925 | 0.5717 | discard (0.011 worse, within noise; +complexity +budget) |
| E62 | drop aux losses w_low=0 & w_coarse_layout=0 | 0.3333 | 1.5433 | 0.5572 | discard (0.070 WORSE — aux losses load-bearing for RMSE/d1; ABS_REL gamed) |
| E63 | FiLM global-audio modulates m16 | 0.3483 | 1.5119 | 0.5700 | discard (0.034 worse, within noise; cross-attn already supplies audio) |
| E64 | learned full-decode 64×128→256×512 | — | — | — | discard (BUDGET BUST 699s/epoch, +150s) |
| **E65** | **drop finest-scale F64 cross-attn** | **0.3422** | **1.4795** | **0.5805** | **KEEP — NEW CHAMPION (comp 2.093; F64 redundant → 549→380s → 9 epochs/deeper anneal; best RMSE+d1; VRAM 30→17.5GB)** |
| E66 | light 128×256 learned decode | 0.3471 | 1.4853 | 0.5743 | discard (0.021 worse, RMSE up — resolution not the bottleneck; audio→depth-limited) |
| E67 | 3rd rsa16b geometry block (affordable) | 0.3399 | 1.4812 | 0.5795 | discard (identical 2.0949; geometry saturates at 2 blocks even with budget) |
| E68 | widen dim 192→256 | 0.3407 | 1.4896 | 0.5766 | discard (0.014 worse; NOT capacity-limited — confirms anneal-limited) |
| E69 | full anneal (epochs 10→9, LR→0) | 0.3425 | 1.4759 | 0.5789 | confirm (tied 2.0948; anneal depth neutral — E65 win was more epochs). kept epochs=10 |
| E70 | learned audio-token positional embeddings | 0.3430 | 1.4752 | 0.5784 | discard (neutral 2.0957; conv encoder already encodes position) |
| E71 | E65 confirmation rerun | 0.3432 | 1.4821 | 0.5783 | confirm (2.1004, |Δ|=0.007 — champion robust, config true comp ~2.097) |
| E72 | iterative ray refinement (2nd cross-attn+geo pass) | 0.3412 | 1.4825 | 0.5799 | discard (neutral 2.0957; single-pass sufficient) |
| E73 | coarse rays attend both coarse+fine audio (kv4+kv3) | 0.3500 | 1.4809 | 0.5807 | discard (neutral 2.0995; fine audio already via F32) |
| E74 | input in-ch 5→3 (drop phase feats) | 0.3532 | 1.4882 | 0.5578 | discard (0.055 WORSE — IPD phase features load-bearing; 5ch optimal) |
| E75 | richer kv projection (Linear→MLP) | 0.3538 | 1.4949 | 0.5741 | discard (0.032 worse; interface not capacity-limited) |
| E76 | flip_aug OFF | 0.3472 | 1.4912 | 0.5732 | discard (0.027 worse all 3 — L/R mirror aug mildly load-bearing) |
| E77 | encoder bottleneck refine on e4 (+4.7M) | 0.3466 | 1.4802 | 0.5783 | discard (neutral 2.1019; encoder not feature-extraction-limited) |
| E78 | 2nd E65 confirmation rerun | 0.3418 | 1.4819 | 0.5793 | confirm (2.0971; 3 runs mean 2.097 ±0.004 — champion solid) |
| E79 | shorter warmup 1→0.5 epoch | 0.3443 | 1.4905 | 0.5759 | discard (0.015 worse; 1-epoch warmup optimal) |
| E80 | SH-basis ray features (use_sh_pe=True) | 0.3450 | 1.4938 | 0.5770 | discard (neutral-worse 2.1119; xyz+Fourier PE sufficient) |
| E81 | mic-position PE ray features (use_mic_pe=True) | running | | | — |

## Current champion & summary (~50 experiments)

Baseline **0.4434 / 1.5907 / 0.5236** → champion **E51 0.3390 / 1.5105 / 0.5637** (comp **2.147**): **ABS_REL −24%, RMSE −5.0%, d1 +4.0 pts**. Noise floor ≈ **0.0045 composite** (E36 rerun) — only Δ>0.005 is real.

**Post-fusion geometry-aware self-attn reopened the frontier at apparent convergence** — geo-aware ray↔ray self-attn on the FUSED coarse layout m16 (post encoder-skip): E50 (1 block, Δ0.027) then E51 (2 blocks, Δ0.015) both big wins. Distinct from E23 (depth on *pre-fusion* tokens, saturated): reasoning geometrically over the *assembled* layout is a fresh, productive axis that BENEFITS from depth. Lesson: keep probing novel architectural ideas even at apparent convergence; and judge only on the fully-annealed final epoch (E51 looked like a loss at epoch 6, won decisively at epoch 7).

**Robust wins (each cleared the noise floor):**
1. **bf16 AMP + batch 32 + cosine anneal** (E0b) — foundation; more epochs/hr + real annealing.
2. **lr 4e-4** (E16) — best honest-metric LR (8e-4→6e-4→4e-4 monotone; 3e-4 U-turns).
3. **Weight EMA, decay 0.995** (E14/E16) — temporal weight average; free; smooths late-training noise.
4. **Coarse 16×32 ray↔ray self-attn** (E22) — layout rays reason jointly; first architectural win.
5. **Geometry-aware bias on that self-attn** (E27) — learned per-head bias on ray-pair cos angular distance.
6. **Gated DPT skips** (E29) — ray features gate per-scale how much encoder detail to admit.
7. **Light edge/gradient loss w_grad=0.05** (E34) — marginal; sharpens boundaries (heavier hurts RMSE-balance).

**Dead ends:** capacity adds (deeper/wider coarse block E23/E25, 2nd audio read E37, mid-scale global attn E24/E26) — saturate or bust the epoch budget; berHu loss (E38–E40) — strong RMSE lever but a flat frontier slide (trades d1/ABS_REL); larger LR/EMA/wd/w_rel/w_grad off-optimum; batch size (E42), coarse-to-fine guidance (E43), global-audio FiLM (E44) — neutral/redundant; larger local-attn window (E41) — too costly.

**Invariant honored throughout:** RayBank ray queries × audio cross-attention (ray-conditioning). Continuing to probe occasional novel ideas from a bank (SwiGLU, log-depth aux loss, attention temperature, EMA warmup) — judged strictly > noise floor.

(E0 fp16 AMP crashed: NaN at epoch 2 → fixed with bf16.)

## Key principles
- **Judge by the honest-weighted composite** (RMSE + d1 dominate; ABS_REL is gamable). Require Δ > **0.005 noise floor** to crown a champion, and judge on the **fully-annealed final epoch** (E51 trailed at ep6, won at ep7).
- **The 1-hour budget is now the binding constraint.** The champion runs ~546s/epoch, right at the 7-epoch ceiling; anything pushing epoch time >~555s drops to 6 epochs → under-anneals → auto-loses (E24/E37/E41/E53). So new wins must be **compute-neutral or cheaper**.
- **The productive axis is ray-conditioned geometric reasoning over the coarse layout**, not loss/schedule tweaks (those slide along the ABS_REL↔RMSE frontier within noise) and not raw capacity (saturates or busts budget).

## Ongoing / next
Mine the geometry-on-fused-layout axis within the budget: cost-neutral variants (drop the now-redundant pre-fusion rsa16? E54), richer geometry features on the post-fusion blocks only, cheaper ways to bring geometric reasoning to finer scales. `results.tsv` is the authoritative append-only log. Loop runs autonomously and indefinitely (see `program.md`).
