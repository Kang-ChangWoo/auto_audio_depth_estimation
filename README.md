# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP planar (cubemap) depth (SoundSpaces).

<!-- RESEARCH:START -->
## Autonomous research state

| | |
|---|---|
| **Mode** | `SYNTHESIZE` — no runs; review evidence, find contradictions, pick the highest-information next question |
| **Active study** | `S0` [new] batvision-reference (*running*) |
| **Research question** | Before any RayDPT work the BatVision U-Net reference must be re-anchored under the PLANAR depth target, and we must know which binaural input representation the reference actually needs. Two questions |
| **Current action** | 2x2 grid on run_base.py, 1h each, sequential on one GPU: {2ch=[L,R], 5ch=[L,R,ILD,cosIPD,sinIPD]} x {use_log True, False}. E0 batvision_2ch_nolog, E1 batvision_2ch_log, E2 batvisio |
| **Latest result** | `E1` batvision_2ch_log: composite **1.8784** (rmse 1.3116, d1 0.5808, abs_rel 0.4211), best epoch 14/26 |
| **Next decision** | Rank by the honest composite. The winning cell becomes the reference number RayDPT must beat. The qualitative figure shows ONE column per channel count, always the NON-LOG variant (operator instructio |
| **Why this mode** | Two grid cells concluded and both produced unexplained results (D1: a pre-registered prediction failed; D2: a budget confound that affects every future run). Rather than spend the remaining GPU on mor |

### Current hypothesis

- **General** — Before any RayDPT work the BatVision U-Net reference must be re-anchored under the PLANAR depth target, and we must know which binaural input representation the reference actually needs. Two questions: (a) how much do the derived interaural cues (ILD, cosIPD, sinIPD) add over the bare [L,R] magnitude pair; (b) does log1p magnitude compression help or hurt.
- **Detailed** — log1p compression equalises the dynamic range between the direct sound and the far-field echo tail, so it should help a model that must read weak late reflections to place distant surfaces -> log wins, and it should help the 2ch stack MORE than the 5ch stack (in 5ch, ILD already supplies a log-domain ratio, so part of the compression benefit is redundant). Adding ILD+IPD gives explicit interaural level/phase, which is where azimuth and hence per-direction depth structure lives -> 5ch beats 2ch at both log settings.
- **Implementation note** — 2x2 grid on run_base.py, 1h each, sequential on one GPU: {2ch=[L,R], 5ch=[L,R,ILD,cosIPD,sinIPD]} x {use_log True, False}. E0 batvision_2ch_nolog, E1 batvision_2ch_log, E2 batvision_5ch_nolog, E3 batvision_5ch_log. Driver: utils/run_batvision_grid.sh.

### Research portfolio

| Idea | Mechanism family | Causal distance | Target bottleneck | Status | Next test |
|---|---|---|---|---|---|
| `I1` | acoustic-representation / temporal resolution | far | time-of-flight quantisation in the input representation | backlog | READY. Probe `run_base.py --stft-hop 40` vs the hop-160 control once the batvision grid fr |
| `I3` | training-optimization | near | the 1h wall-clock budget is spent on epochs that make the model worse | backlog | queue after the RayDPT planar re-anchor (E4); this is a confound affecting EVERY future ru |

### Open discrepancies

*Unexplained observations are research assets, not noise.*

- **`D1`** — log1p magnitude compression is a no-op at 2ch: E1 (log) composite 1.8784 vs E0 (nolog) 1.8854, delta 0.0070 < sigma 0.008. The two runs SPLIT the underlying metrics -- log wins rmse (1.3116 vs 1.3186) and d1 (0.5808 vs 0.5785), nolog wins abs_rel (0.4143 vs 0.4211).
  <br/>*Why it matters:* The pre-registered detailed hypothesis predicted log would help MORE at 2ch than at 5ch, because in the 5ch stack ILD already supplies a log-domain ratio. At 2ch it does not help at all, so that half of the prediction is unsupported. A metric split with a sub-sigma delta is the signature of noise, not of a mechanism.
- **`D3`** — E2 (5ch nolog, composite 1.8646) beats both 2ch cells (1.8854, 1.8784) by 0.014-0.021 -- above the sigma~0.008 floor -- but the win comes ENTIRELY from d1 (0.5938 vs 0.5808/0.5785). Its RMSE is actually WORSE than E1's (1.3207 vs 1.3116) and its ABS_REL is the worst of the three (0.4460 vs 0.4143/0.4211).
  <br/>*Why it matters:* The composite weights d1 at 1/0.46, so a d1 gain of 0.013 alone buys 0.028 of composite -- enough to swamp the RMSE regression. So 'derived interaural cues help' is true ONLY in the d1 sense: the cues make more pixels land within +-25% of truth, while the typical squared error gets slightly larger. That is the profile of a mechanism that improves DIRECTIONAL/angular assignment (which is exactly what ILD and IPD encode: azimuth) without improving absolute RANGE. It is not evidence that the cues help distance estimation.
- **`D2`** — Both 2ch cells peak at epoch 14 of 26 and both peak at exactly 2400.3 MB VRAM.
  <br/>*Why it matters:* The overfitting turn and the memory envelope are properties of the architecture + schedule, NOT of the input representation. This makes epoch count a CONFOUND for every comparison run under the fixed wall-clock budget: any change that slows an epoch silently reduces the epochs that fit, and is penalised for reasons unrelated to its mechanism.

### Recent decisions

| When | Mode | Event | Note |
|---|---|---|---|
| 2026-07-10T03:28 | `synthesize` | discrepancy_recorded | D3: E2's composite win over the 2ch cells is entirely a d1 win (0.5938 vs 0.5808); its RMSE (1.3207) is WORSE than E1's (1.3116) a |
| 2026-07-10T03:26 | `synthesize` | candidate_dropped | I2 (decoder output resolution) REFUTED by an oracle diagnostic costing zero GPU. A PERFECT predictor at RayDPT's 64x128 decode res |
| 2026-07-10T03:26 | `synthesize` | experiment_completed | batvision 5ch nolog: composite 1.8646 (rmse 1.3207, d1 0.5938, abs_rel 0.4460), best epoch 13/25. Beats both 2ch cells (1.8854, 1. |
| 2026-07-10T03:24 | `synthesize` | study_opened | S1 (queued, exploit): E4 re-anchors RayDPT (my model) under the planar target. Prerequisite for every RayDPT improvement; nothing  |
| 2026-07-10T03:24 | `synthesize` | study_opened | S2 (queued, explore): I1 temporal-resolution probe on the CHEAPEST parent (batvision, not the champion). Control is E2 (batvision_ |
| 2026-07-10T03:19 | `synthesize` | infrastructure | Upgrade-plan section 15 (audio representation search-space): STFT analysis window moved from prepare.py module constants into cfg. |
| 2026-07-10T03:20 | `synthesize` | infrastructure | Serial scored-evaluation lock added (utils/evallock.py). Our composite has no runtime term, but TIME_BUDGET is wall-clock, so over |
| 2026-07-10T03:12 | `synthesize` | idea_added | Temporal resolution of the input (hop 160 -> 40). Causally FAR from the current decoder/attention lineage, grounded in sensing phy |

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
