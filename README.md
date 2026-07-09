# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP planar (cubemap) depth (SoundSpaces).

<!-- RESEARCH:START -->
## Autonomous research state

| | |
|---|---|
| **Mode** | `EXPLOIT` — adaptive HPO ladder 3 -> 5 -> 7 -> 10, each step justified by evidence -> PASS / FAIL |
| **Active study** | `S2` [new] acoustic-representation (*running*) |
| **Research question** | Depth from echoes is a time-of-flight measurement: a surface at distance d returns its echo at t=2d/c. If the input representation cannot resolve t, no decoder can resolve d. The analysis window, not  |
| **Current action** | E5 batvision_5ch_win400_hop40, E6 batvision_5ch_win64_hop16, on run_base.py (cheapest parent). Control = E2. |
| **Latest result** | *(no scored run in this study yet)* |
| **Next decision** | Per D3 the gain must appear in RMSE (range), not d1 (angle) -- the interaural cues already own d1. Neither arm improves RMSE -> DROP I1. Both improve equally -> the gain is sampling density/capacity,  |
| **Why this mode** | RayDPT's throughput (I8) now gates every RayDPT judgement, and the user asked for a run that reaches 25 epochs. Three output-equivalent speedups are in (dead-tail deletion, unfold-free attention, bf16 |

### Current hypothesis

- **General** — Depth from echoes is a time-of-flight measurement: a surface at distance d returns its echo at t=2d/c. If the input representation cannot resolve t, no decoder can resolve d. The analysis window, not the hop, sets that resolution.
- **Detailed** — win=400 smears an echo over c*win/(2*sr)=1.417 m of one-way depth -- essentially the achieved RMSE of 1.3088 m. Arm A (win 400, hop 40) raises sampling density while leaving the smear unchanged; arm B (win 64, hop 16) cuts the smear to 0.227 m, paying frequency resolution. Only B should improve RMSE if temporal resolution binds.
- **Implementation note** — E5 batvision_5ch_win400_hop40, E6 batvision_5ch_win64_hop16, on run_base.py (cheapest parent). Control = E2.

### Research portfolio

| Idea | Mechanism family | Causal distance | Target bottleneck | Status | Next test |
|---|---|---|---|---|---|
| `I1` | acoustic-representation / temporal resolution | far | time-of-flight quantisation in the input representation | inconclusive | HOLD. Do not run more I1 arms. I10 (bilinear at hop=160) isolates the staircase; interpret |
| `I3` | training-optimization | near | the 1h wall-clock budget is spent on epochs that make the model worse | backlog | queue after the RayDPT planar re-anchor (E4); this is a confound affecting EVERY future ru |
| `I5` | ray conditioning / encoder-decoder correspondence | mid | RayDPT's DPT skip connections impose a FALSE spatial correspondence between the spectrogram's axes and the ERP's axes | inconclusive | none. Do not spend GPU on the skip ablation on this rationale. Revive only with an indepen |
| `I6` | depth objective design | mid | the objective devotes most of its gradient to low-frequency terms, so the model may be trained to be blurry | backlog | expose w_coarse_layout / w_low as CLI flags, then queue the ablation after the I1 arms. |
| `I7` | sensing physics / angular resolution | far | two microphones may fundamentally under-determine high azimuthal frequencies | backlog | none directly -- I7 is decided by I6's outcome. Do NOT call this a task ceiling; it is a s |
| `I8` | throughput / training-optimization | near | RayDPT is COMPUTE-STARVED under the fixed 1-hour wall-clock budget | probing | bench_raydpt.py is queued behind the eval_lock; E7/E8 run in utils/run_queue4.sh. |
| `I10` | acoustic-representation / interpolation | mid | the nearest-neighbour resize in _features() turns the time axis into a coarse staircase | probing | queued in utils/run_queue3.sh after the I6/I7 discriminator. |

### Open discrepancies

*Unexplained observations are research assets, not noise.*

- **`D2`** — Both 2ch cells peak at epoch 14 of 26 and both peak at exactly 2400.3 MB VRAM.
  <br/>*Why it matters:* The overfitting turn and the memory envelope are properties of the architecture + schedule, NOT of the input representation. This makes epoch count a CONFOUND for every comparison run under the fixed wall-clock budget: any change that slows an epoch silently reduces the epochs that fit, and is penalised for reasons unrelated to its mechanism.
- **`D5`** — E4 (RayDPT) fitted only 5 epochs in the 1-hour budget (713.5 s/epoch) versus batvision's 25 (~130 s/epoch). Its best checkpoint is the LAST epoch and all val metrics were still improving monotonically. Peak VRAM 16.2 GB vs 2.4 GB.
  <br/>*Why it matters:* E4's composite 2.0471 vs E3's 1.8567 CANNOT be read as 'RayDPT is the worse model'. One model converged and then overfit for 12 epochs; the other never reached convergence. Under a wall-clock budget, throughput is silently part of the score. Every RayDPT-vs-batvision statement in this phase must carry epochs_ran, and no RayDPT mechanism can be fairly judged until RayDPT can converge inside the budget.
- **`D6`** — I1's CONTROL arm improved. E5 (win400 hop40) cut RMSE by 0.0227 and the composite by 0.0132 (above sigma) versus E2, despite leaving the analysis window -- and therefore the temporal smear of 1.417 m -- completely unchanged.
  <br/>*Why it matters:* The arm was designed to isolate 'sampling density' from 'resolution' and to show nothing. It showed something, and in RMSE, the very metric reserved as evidence FOR the resolution mechanism. A Nyquist check says hop=160 already samples the window's envelope at 1.2x Nyquist, so arm A added ~no information. Something other than information improved range accuracy.
- **`D7`** — The two I1 arms improved the composite through OPPOSITE metrics. Arm A (density only, smear unchanged): rmse -0.0227, d1 -0.0019. Arm B (6.2x finer smear): rmse -0.0093, d1 +0.0067 -- and B's RMSE is worse than A's despite B having vastly better temporal resolution.
  <br/>*Why it matters:* If temporal resolution set range accuracy, B should own RMSE. It does not; A does, and A did not change resolution at all. Meanwhile B, which also sacrifices frequency resolution (win 400 -> 64), buys ANGLE. That inverts the physical story: sharper transients seem to help azimuth cues (ILD/IPD are read across frequency and time), while range accuracy responds to something in the sampling/interpolation of the time axis.

### Recent decisions

| When | Mode | Event | Note |
|---|---|---|---|
| 2026-07-10T08:33 | `exploit` | idea_added | VALIDATED: e5..e8 deletion is output bit-identical (max\|diff\|=0.0). RayDPT 24.44M -> 7.66M params. |
| 2026-07-10T08:33 | `exploit` | mode_changed | RayDPT's throughput (I8) now gates every RayDPT judgement, and the user asked for a run that reaches 25 epochs. Three output-equiv |
| 2026-07-10T07:36 | `explore` | discrepancy_recorded | D7: the two I1 arms improved through OPPOSITE metrics, inverting the physical story. The only variable monotone with the composite |
| 2026-07-10T07:36 | `explore` | experiment_completed | I1 arm B (win64 hop16, 6.2x finer smear): composite 1.8379, best batvision so far. But the gain is d1 (+0.0067) and abs_rel, NOT r |
| 2026-07-10T06:36 | `explore` | discrepancy_recorded | D6: I1's CONTROL arm improved. E5 (win400 hop40) cut RMSE 0.0227 and composite 0.0132 (above sigma) with the temporal smear UNCHAN |
| 2026-07-10T06:34 | `explore` | experiment_completed | I1 arm A (win400 hop40, density only, smear UNCHANGED at 1.417m): composite 1.8514 vs control E2 1.8646, delta +0.0132 ABOVE sigma |
| 2026-07-10T05:33 | `explore` | discrepancy_recorded | D5: RayDPT is compute-starved. Under a wall-clock budget throughput is silently part of the score. 64x128 attention (lsa64+cross64 |
| 2026-07-10T05:33 | `explore` | experiment_completed | RayDPT planar anchor: composite 2.0471 (rmse 1.3987, d1 0.5423), but ONLY 5 epochs fit (713.5s/ep vs batvision 130s). Best = last  |

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
