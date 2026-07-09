# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP planar (cubemap) depth (SoundSpaces).

<!-- RESEARCH:START -->
## Autonomous research state

| | |
|---|---|
| **Mode** | `EXPLOIT` — adaptive HPO ladder 3 -> 5 -> 7 -> 10, each step justified by evidence -> PASS / FAIL |
| **Active study** | `S1` [new] raydpt-baseline (*running*) |
| **Research question** | RayDPT (my ray-conditioned model) must be re-anchored under the PLANAR target before any of its mechanisms can be judged. Nothing measured before commit 87b3047 is comparable, so the model currently h |
| **Current action** | E4: python train.py --mode train (defaults). Queued in utils/run_queue.sh; started 04:28. |
| **Latest result** | *(no scored run in this study yet)* |
| **Next decision** | This is an ANCHOR, not a competition: E4 is recorded whatever it scores. Report which metrics RayDPT wins/loses vs E3 separately (per S0, d1 is the cue-driven metric and rmse the range metric). If Ray |
| **Why this mode** | S0 concluded PASS: the reference is anchored at composite 1.8567 and the representation question is answered (cues->angle, log->range, additive). Now EXPLOIT the prerequisite: E4 re-anchors RayDPT (my |

### Current hypothesis

- **General** — RayDPT (my ray-conditioned model) must be re-anchored under the PLANAR target before any of its mechanisms can be judged. Nothing measured before commit 87b3047 is comparable, so the model currently has NO valid baseline in this phase and no RayDPT improvement can be attributed.
- **Detailed** — Run train.py unchanged (5ch log, flip-aug, no TTA) for one wall-clock hour. This produces the parent against which decoder, representation and optimisation mechanisms will be compared, and gives the first honest RayDPT-vs-batvision number under the planar target. The reference to beat is E3 = 1.8567.
- **Implementation note** — E4: python train.py --mode train (defaults). Queued in utils/run_queue.sh; started 04:28.

### Research portfolio

| Idea | Mechanism family | Causal distance | Target bottleneck | Status | Next test |
|---|---|---|---|---|---|
| `I1` | acoustic-representation / temporal resolution | far | time-of-flight quantisation in the input representation | probing | RUNNING in the queue after E4: arm A = batvision_5ch_win400_hop40, arm B = batvision_5ch_w |
| `I3` | training-optimization | near | the 1h wall-clock budget is spent on epochs that make the model worse | backlog | queue after the RayDPT planar re-anchor (E4); this is a confound affecting EVERY future ru |
| `I5` | ray conditioning / encoder-decoder correspondence | mid | RayDPT's DPT skip connections impose a FALSE spatial correspondence between the spectrogram's axes and the ERP's axes | backlog | run the zero-GPU azimuthal-FFT diagnostic on the E3 batvision predictions while E4 trains. |

### Open discrepancies

*Unexplained observations are research assets, not noise.*

- **`D2`** — Both 2ch cells peak at epoch 14 of 26 and both peak at exactly 2400.3 MB VRAM.
  <br/>*Why it matters:* The overfitting turn and the memory envelope are properties of the architecture + schedule, NOT of the input representation. This makes epoch count a CONFOUND for every comparison run under the fixed wall-clock budget: any change that slows an epoch silently reduces the epochs that fit, and is penalised for reasons unrelated to its mechanism.
- **`D4`** — Predicted depth maps in out/display/qualitative.png are low-frequency blobs showing quasi-periodic banding along azimuth, while the GT is sharp and piecewise-constant. The 5ch predictions carry visibly more azimuthal structure than the 2ch ones.
  <br/>*Why it matters:* The input's width axis is STFT TIME with only 18 distinct frames, nearest-upsampled 28.4x into 512 columns -- a staircase. If the predicted banding period matches that staircase, the model is reproducing an artefact of the interpolation rather than scene geometry, and the nearest-neighbour upsample in _features() (editable research logic) is itself a defect. The extra 5ch structure is consistent with S0's finding that interaural cues buy angular assignment.

### Recent decisions

| When | Mode | Event | Note |
|---|---|---|---|
| 2026-07-10T04:32 | `exploit` | idea_added | RayDPT's DPT skips add encoder features to the ray grid by pixel index, equating (frequency,time) with (elevation,azimuth) -- a fa |
| 2026-07-10T04:30 | `exploit` | hypothesis_concluded | S0 PASS. 2x2 dissociation: interaural cues buy d1 (angle, +0.015), log1p buys rmse (range, -0.010); effects ADDITIVE. Predicted lo |
| 2026-07-10T04:30 | `exploit` | experiment_completed | batvision 5ch log: composite 1.8567 (rmse 1.3088, d1 0.5949, abs_rel 0.4517), best ep13/25. Best of the 2x2; reference champion. |
| 2026-07-10T03:42 | `synthesize` | discrepancy_recorded | Failed instrument: a training-free NN-retrieval probe intended to test I1 cheaply is INVALID -- its retrieval (rmse 2.20) is worse |
| 2026-07-10T03:42 | `synthesize` | direction_changed | I1 probe REDESIGNED before spending GPU. The STFT's temporal resolution is set by the analysis WINDOW, not the hop: win=400 smears |
| 2026-07-10T03:28 | `synthesize` | discrepancy_recorded | D3: E2's composite win over the 2ch cells is entirely a d1 win (0.5938 vs 0.5808); its RMSE (1.3207) is WORSE than E1's (1.3116) a |
| 2026-07-10T03:26 | `synthesize` | candidate_dropped | I2 (decoder output resolution) REFUTED by an oracle diagnostic costing zero GPU. A PERFECT predictor at RayDPT's 64x128 decode res |
| 2026-07-10T03:26 | `synthesize` | experiment_completed | batvision 5ch nolog: composite 1.8646 (rmse 1.3207, d1 0.5938, abs_rel 0.4460), best epoch 13/25. Beats both 2ch cells (1.8854, 1. |

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
    subgraph MY["current — RayDPT (my model)"]
        direction TB
        A1["Binaural echo waveform (2ch)"] --> A2["STFT → named cue stack (in_ch)<br/>logL/L · logR/R · ILD · cosIPD · sinIPD"]
        A2 --> A3["UNet8 encoder<br/>256x512 → 1x2 · skips e2/e3/e4"]
        A3 --> A4["RayBank ray queries ×<br/>audio cross-attention (scales 16/32/64)"]
        A4 --> A5["DPT fusion +<br/>local spherical window attention"]
        A5 --> A6["Sigmoid head → ERP planar depth<br/>256x512, [0,1] × max_depth"]
    end
    subgraph REF["batvision (reference)"]
        direction TB
        B1["Binaural echo waveform (2ch)"] --> B2["STFT → magnitude cue stack (in_ch)"]
        B2 --> B3["UNet8 encoder<br/>256x512 → 1x2 · skips"]
        B3 --> B4["ConvTranspose decoder + skips"]
        B4 --> B5["Sigmoid head → ERP planar depth"]
    end
    MY ~~~ REF
```
