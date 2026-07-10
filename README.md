# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP planar (cubemap) depth (SoundSpaces).

<!-- RESEARCH:START -->
## Autonomous research state

| | |
|---|---|
| **Mode** | `VERIFY` — clean reimplementation on the correct parent -> standalone run -> bounded HPO -> PASS / FAIL |
| **Active study** | `S4` [refine] raydpt-capacity (*running*) |
| **Research question** | E9 is a COMPOUND change, so its deficit against the reference cannot be attributed. Before asking whether ray-conditioning helps, we must know which part of the capacity we removed was load-bearing -- |
| **Current action** | E10 raydpt_e10_d32L2_b64: --decode-scale 32 --ray-cross-layers 2 --batch-size 64 --lr 1.2e-3 --amp bf16 --epochs 16. |
| **Latest result** | *(no scored run in this study yet)* |
| **Next decision** | Judge on d1 first, not the composite: D9 says the gap is angular. If d1 recovers materially toward 0.5949, the cross-layer cut was load-bearing. If d1 does not move, the cut was free and the deficit l |
| **Why this mode** | E9 established convergence but is a compound change. Before any claim about ray-conditioning, ablate the amputation: restore the cross-attention depth and see whether the d1 deficit (D9) is ours or th |

### Current hypothesis

- **General** — E9 is a COMPOUND change, so its deficit against the reference cannot be attributed. Before asking whether ray-conditioning helps, we must know which part of the capacity we removed was load-bearing -- otherwise we would be judging a mechanism by an amputation we chose ourselves.
- **Detailed** — E9 cut both the decode scale (64->32) and the cross-attention depth (2->1 layers per ray scale). The audio<->ray cross-attention is where per-ray direction information enters the model, so halving it is the change most likely to have cost the d1 (angular) deficit seen in D9. Restoring ray_cross_layers=2 at decode 32 still fits the budget (measured 216.5 s/epoch = 16.6 epochs/h).
- **Implementation note** — E10 raydpt_e10_d32L2_b64: --decode-scale 32 --ray-cross-layers 2 --batch-size 64 --lr 1.2e-3 --amp bf16 --epochs 16.

### Research portfolio

| Idea | Mechanism family | Causal distance | Target bottleneck | Status | Next test |
|---|---|---|---|---|---|
| `I1` | acoustic-representation / temporal resolution | far | time-of-flight quantisation in the input representation | inconclusive | HOLD. Do not run more I1 arms. I10 (bilinear at hop=160) isolates the staircase; interpret |
| `I3` | training-optimization | near | the 1h wall-clock budget is spent on epochs that make the model worse | backlog | queue after the RayDPT planar re-anchor (E4); this is a confound affecting EVERY future ru |
| `I5` | ray conditioning / encoder-decoder correspondence | mid | RayDPT's DPT skip connections impose a FALSE spatial correspondence between the spectrogram's axes and the ERP's axes | inconclusive | none. Do not spend GPU on the skip ablation on this rationale. Revive only with an indepen |
| `I7` | sensing physics / angular resolution | far | two microphones may fundamentally under-determine high azimuthal frequencies | candidate | Do not chase high-frequency power as a goal. Re-test the observability claim once RayDPT c |
| `I10` | acoustic-representation / interpolation | mid | the nearest-neighbour resize in _features() turns the time axis into a coarse staircase | inconclusive | deferred confirm: run `--feat-interp bilinear --stft-hop 40` after the RayDPT throughput s |

### Open discrepancies

*Unexplained observations are research assets, not noise.*

- **`D2`** — Both 2ch cells peak at epoch 14 of 26 and both peak at exactly 2400.3 MB VRAM.
  <br/>*Why it matters:* The overfitting turn and the memory envelope are properties of the architecture + schedule, NOT of the input representation. This makes epoch count a CONFOUND for every comparison run under the fixed wall-clock budget: any change that slows an epoch silently reduces the epochs that fit, and is penalised for reasons unrelated to its mechanism.
- **`D7`** — The two I1 arms improved the composite through OPPOSITE metrics. Arm A (density only, smear unchanged): rmse -0.0227, d1 -0.0019. Arm B (6.2x finer smear): rmse -0.0093, d1 +0.0067 -- and B's RMSE is worse than A's despite B having vastly better temporal resolution.
  <br/>*Why it matters:* If temporal resolution set range accuracy, B should own RMSE. It does not; A does, and A did not change resolution at all. Meanwhile B, which also sacrifices frequency resolution (win 400 -> 64), buys ANGLE. That inverts the physical story: sharper transients seem to help azimuth cues (ILD/IPD are read across frequency and time), while range accuracy responds to something in the sampling/interpolation of the time axis.
- **`D8`** — E6 holds 29.7% LESS high-frequency azimuthal power than E2 (0.0232 vs 0.0331) yet has a BETTER d1 (0.6005 vs 0.5938). Separately, removing 58% of the gradient (E7 vs E3) barely changed the spectrum or the composite.
  <br/>*Why it matters:* It breaks the assumption -- mine, unstated until now -- that d1 improves because predictions get sharper. d1 counts pixels within +-25% of truth, and a well-centred smooth field beats a mis-placed sharp one. So the low-pass character of these models may be largely IRRELEVANT to the metric, and 'restore high frequencies' is probably the wrong research goal.
- **`D9`** — With BOTH models converged, RayDPT (E9, 1.9308) trails batvision (E3, 1.8567) by 0.0741 -- and 0.0691 of that gap is d1 alone (0.5631 vs 0.5949). RMSE is essentially tied (1.3195 vs 1.3088, contributing 0.0067).
  <br/>*Why it matters:* d1 is the ANGULAR metric: S0 established that the interaural cues (ILD, IPD), which encode azimuth, buy d1 while log1p compression buys rmse. So the ray-conditioned model -- whose entire premise is that depth should be decoded per ray DIRECTION -- is worse at direction than a plain encoder-decoder with a 1x2 bottleneck, while matching it on range. The mechanism is losing exactly where it claims to win.

### Recent decisions

| When | Mode | Event | Note |
|---|---|---|---|
| 2026-07-10T10:58 | `verify` | candidate_dropped | REFUTED, zero GPU: the model WITHOUT the 1x2 waist (RayDPT E9) is SMOOTHER than the one with it (batvision E3) -- 0.0179 vs 0.0290 |
| 2026-07-10T10:55 | `verify` | discrepancy_recorded | D9: the ray-conditioned model loses to a plain U-Net almost entirely on d1 (angle), while tying on rmse (range). It is losing exac |
| 2026-07-10T10:55 | `verify` | experiment_completed | CONVERGED RayDPT: composite 1.9308, 21 epochs, best at ep18/21. Beats starved E4 (2.0471) by 0.1162 = 14 sigma. D5 confirmed: RayD |
| 2026-07-10T09:51 | `exploit` | direction_changed | Pure speed exhausted at 1.4x (500 s/epoch, 7.2 epochs). Reaching 25 epochs needs a CAPACITY cut, recorded as such: decode_scale 32 |
| 2026-07-10T09:42 | `exploit` | experiment_completed | I10 discriminator (bilinear at hop160, information identical to E2): rmse -0.0119 (52% of E5's gain, predicted direction) but d1 - |
| 2026-07-10T08:43 | `exploit` | discrepancy_recorded | Throughput reality check: bench measures 519 s/epoch at best (batch 48, bf16) = 6.9 epochs in the budget, versus 144 s/epoch neede |
| 2026-07-10T08:43 | `exploit` | experiment_completed | batvision 5ch log, aux losses ZEROED: composite 1.8613 vs E3's 1.8567 (delta +0.0046, below sigma). Azimuthal high-frequency power |
| 2026-07-10T08:40 | `exploit` | discrepancy_recorded | D8: E6 has 29.7% LESS high-frequency power than E2 yet BETTER d1. Sharpness and d1 are not the same thing -- a well-centred smooth |

*Updated by `python utils/report.py research`. Champion: none yet.*
<!-- RESEARCH:END -->

**Reference model** = BatVision U-Net (`base/`, plain pix2pix encoder→decoder, trained by
`run_base.py`). **My model** = the ray-conditioned RayDPT (`train.py`), iterated to beat the
reference under the same fixed split / target / metric / selection composite.

**Input representation** — named binaural cues, each on/off, plus a `use_log` switch
(`prepare.build_channel_names`): `logL/L, logR/R, ILD, cosIPD, sinIPD`. Default = all five,
`use_log=True` → the 5ch `[logL,logR,ILD,cosIPD,sinIPD]` stack.

## Visual results

Held-out val scenes — `RGB | GT depth | batvision (2ch) | batvision (5ch) | current (my model)`.
The batvision reference gets exactly one column per channel count, always the **non-log** variant;
the log variants are still trained and logged to `out/results.tsv`. "my model" fills in as improved
RayDPT checkpoints are found. RGB is unavailable in the simplified dataset.

![qualitative depth comparison](out/display/qualitative.png)

Performance vs experiment (honest composite `rmse/1.6 + (1-d1)/0.46 + 0.35·abs_rel`, lower = better;
running best highlighted):

![performance progress](out/display/score_progress.png)

*Regenerate: `conda activate ss && python utils/report.py all`.*

## Results

<!-- RESULTS:START -->
| # | commit | ABS_REL | RMSE | d1 | composite | status | description |
|---|---|---|---|---|---|---|---|
| 1 | `209c6e8` | 0.4143 | 1.3186 | 0.5785 | 1.8854 | keep | E0 batvision U-Net 2ch [L,R] nolog, planar target, 26ep |
| 2 | `209c6e8` | 0.4211 | 1.3116 | 0.5808 | 1.8784 | keep | E1 batvision U-Net 2ch [logL,logR] log, planar target, 26ep |
| 3 | `209c6e8` | 0.4460 | 1.3207 | 0.5938 | 1.8646 | keep | E2 batvision U-Net 5ch nolog, planar target, 25ep |
| 4 | `209c6e8` | 0.4517 | 1.3088 | 0.5949 | 1.8567 | keep | E3 batvision U-Net 5ch log, planar target, 25ep |
| 5 | `9dd3bce` | 0.5081 | 1.3987 | 0.5423 | 2.0470 | keep | E4 RayDPT planar anchor, 5ep ONLY (713s/ep), best=last ep, undertrained |
| 6 | `b9c2f71` | 0.4371 | 1.2980 | 0.5919 | 1.8514 | keep | E5 batvision 5ch nolog win400 hop40 (I1 arm A: density only), 26ep |
| 7 | `b9c2f71` | 0.4279 | 1.3114 | 0.6005 | 1.8379 | keep | E6 batvision 5ch nolog win64 hop16 (I1 arm B: true resolution), 25ep |
| 8 | `b9c2f71` | 0.4540 | 1.3155 | 0.5951 | 1.8613 | keep | E7 batvision 5ch log, aux losses ZEROED (I6 vs I7 discriminator) |
| 9 | `7fae910` | 0.4470 | 1.3088 | 0.5900 | 1.8658 | keep | E8 batvision 5ch nolog win400 hop160 BILINEAR resize (I10: staircase discriminator) |
| 10 | `bb9692d` | 0.4468 | 1.3195 | 0.5631 | 1.9309 | keep | E9 RayDPT decode32 xlayers1 batch64 lr1.2e-3 bf16 ep25 (S3: does a CONVERGED RayDPT beat a starved one?) |
<!-- RESULTS:END -->

## Progression (composite, lower = better)

| phase | best | note |
|---|---|---|
| 2026-June (archived) | ~2.030 | multi-res STFT + interaural coherence + TTA |
| 2026-July (this) | — | BatVision reference + named-cue inputs + fixed coarse/low loss target |

## Network flowchart

Two separate top-down networks — **current** (RayDPT, my model) on top, the **BatVision reference**
below:

```mermaid
flowchart TD
    subgraph MY["current — RayDPT (my model) · 24.44M params, of which 16.78M are DEAD"]
        direction TB
        A1["Binaural echo waveform (2ch)<br/>cut = 2·10m/c → 2823 samples · FIXED"] --> A2["STFT(nfft 512, hop 160, win 400)<br/>→ (F=257, T=18) → resize to 256×512<br/>height=FREQUENCY · width=TIME"]
        A2 --> A2b["named cue stack (in_ch)<br/>logL/L · logR/R · ILD · cosIPD · sinIPD"]
        A2b --> A3["UNet8 encoder — only e1..e4 run<br/>256×512 → e2 64×128 · e3 32×64 · e4 16×32"]
        A3 -.->|"e5..e8 built but never called<br/>16.78M params · no gradient (I9)"| A3d["(dead tail)"]
        A3 --> A4["RayBank ray queries Q16/Q32/Q64<br/>× GLOBAL audio cross-attention<br/>(kv from e4=512 tok, e3=2048 tok)"]
        A3 --> A4s["DPT skips se2/se3/se4<br/>added by PIXEL INDEX (I5: parked)"]
        A4 --> A5["DPT fusion + local spherical<br/>window attention (32×64 win5, 64×128 win3)<br/>≈57% of forward cost (I8)"]
        A4s --> A5
        A5 --> A6["Sigmoid head at 64×128<br/>→ bilinear ×4 → ERP planar depth 256×512"]
    end
    subgraph REF["batvision (reference) — plain pix2pix unet_256, 54.41M params"]
        direction TB
        B1["Binaural echo waveform (2ch)<br/>same fixed cut"] --> B2["same STFT → named cue stack (in_ch)"]
        B2 --> B3["UNet8 encoder<br/>256×512 → 1×2 (all 8 downs used)"]
        B3 --> B4["ConvTranspose decoder + full skips<br/>decodes to full 256×512"]
        B4 --> B5["Sigmoid head → ERP planar depth"]
    end
    MY ~~~ REF
```

Both models consume the *same* input tensor and are trained by the same `composite_loss`
(`1.0·dense_MAE + 1.0·coarse_layout(16×32) + 0.5·low_pass(σ=3)`; the two auxiliaries carry
~58% of the gradient at convergence — see idea `I6`). The target is **planar** (cubemap
perpendicular-Z) ERP depth. The invariant that defines RayDPT is that depth is decoded **per
ray direction** from `RayBank` queries, never regressed from a global bottleneck.
