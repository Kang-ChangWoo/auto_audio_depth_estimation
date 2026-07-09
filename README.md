# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP radial depth (SoundSpaces).

**Reference model** = BatVision U-Net (`base/`, plain pix2pix encoder→decoder, trained by
`run_base.py`). **My model** = the ray-conditioned RayDPT (`train.py`), iterated to beat the
reference under the same fixed split / target / metric / selection composite.

**Input representation** — named binaural cues, each on/off, plus a `use_log` switch
(`prepare.build_channel_names`): `logL/L, logR/R, ILD, cosIPD, sinIPD`. Default = all five,
`use_log=True` → the 5ch `[logL,logR,ILD,cosIPD,sinIPD]` stack.

## Visual results

Held-out val scenes — `RGB | GT depth | batvision | best1 | best2` (best1/best2 fill in as
improved "my model" checkpoints are found; RGB is unavailable in the simplified dataset).

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
        A5 --> A6["Sigmoid head → ERP radial depth<br/>256x512, [0,1] × max_depth"]
    end
    subgraph REF["batvision (reference)"]
        direction TB
        B1["Binaural echo waveform (2ch)"] --> B2["STFT → magnitude cue stack (in_ch)"]
        B2 --> B3["UNet8 encoder<br/>256x512 → 1x2 · skips"]
        B3 --> B4["ConvTranspose decoder + skips"]
        B4 --> B5["Sigmoid head → ERP radial depth"]
    end
    MY ~~~ REF
```
