# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP planar (cubemap) depth (SoundSpaces).

<!-- RESEARCH:START -->
## Autonomous research state

| | |
|---|---|
| **Mode** | `VERIFY` — clean reimplementation on the correct parent -> standalone run -> bounded HPO -> PASS / FAIL |
| **Active study** | `G0` [new] echo-delay-volume (*running*) |
| **Research question** | Depth from echoes is a time-of-flight measurement: a surface at depth d returns its echo at t = 2d/c. A network that regresses a scalar depth must discover that correspondence from data; a network who |
| **Current action** | E24 = defaults + --depth-volume True (control E21). E25 = champion architecture + depth volume (control E23). E26 = confirm draw. |
| **Latest result** | `E24` raydpt_e24_echodelay: composite **1.8987** (rmse 1.3183, d1 0.5733, abs_rel None), best epoch 23/24 |
| **Next decision** | The pre-registered falsification (far deciles must improve) PASSED. Crowning still requires (a) the mechanism measured on the champion architecture, since E24 carried knobs that cost 0.0137, and (b) a |
| **Why this mode** | I19's pre-registered prediction was confirmed -- the first in the project. Verify it on the correct parent and with a second draw before crowning. |

### Current hypothesis

- **General** — Depth from echoes is a time-of-flight measurement: a surface at depth d returns its echo at t = 2d/c. A network that regresses a scalar depth must discover that correspondence from data; a network whose architecture encodes it does not. Put the physics in the structure, not in the loss.
- **Detailed** — The encoder's width axis IS time -- spec is (freq 256, time 512), so e3 is (freq 32, time 64) and its columns are depth hypotheses spanning 0.08-9.92 m. Each ray attends over FREQUENCY within one time column per hypothesis, a small MLP scores the hypotheses, and a softmax over the DEPTH axis gives p(d|ray); depth = soft-argmax over echo delay. It also replaces the sigmoid coarse head that saturated in D11.
- **Implementation note** — E24 = defaults + --depth-volume True (control E21). E25 = champion architecture + depth volume (control E23). E26 = confirm draw.

### Research portfolio

| Idea | Mechanism family | Causal distance | Target bottleneck | Status | Next test |
|---|---|---|---|---|---|
| `I1` | acoustic-representation / temporal resolution | far | time-of-flight quantisation in the input representation | inconclusive | HOLD. Do not run more I1 arms. I10 (bilinear at hop=160) isolates the staircase; interpret |
| `I3` | training-optimization | near | the 1h wall-clock budget is spent on epochs that make the model worse | backlog | queue after the RayDPT planar re-anchor (E4); this is a confound affecting EVERY future ru |
| `I5` | ray conditioning / encoder-decoder correspondence | mid | RayDPT's DPT skip connections impose a FALSE spatial correspondence between the spectrogram's axes and the ERP's axes | inconclusive | none. Do not spend GPU on the skip ablation on this rationale. Revive only with an indepen |
| `I7` | sensing physics / angular resolution | far | two microphones may fundamentally under-determine high azimuthal frequencies | candidate | Do not chase high-frequency power as a goal. Re-test the observability claim once RayDPT c |
| `I10` | acoustic-representation / interpolation | mid | the nearest-neighbour resize in _features() turns the time axis into a coarse staircase | inconclusive | deferred confirm: run `--feat-interp bilinear --stft-hop 40` after the RayDPT throughput s |
| `I14` | ray conditioning / audio token routing | mid | far-field rays cannot see the late, weak echo that carries distance | probing | E16 (control) then E15b (treatment), both at lr 6e-4. Pre-registered falsification unchang |
| `I19` | ray conditioning / physically-structured decoding | far | the model must LEARN that echo delay encodes depth, and it fails to, collapsing far surfaces toward the median | candidate | E25 (champion config + depth volume, control E23 = 1.8962) and E26 (confirm draw). |

### Open discrepancies

*Unexplained observations are research assets, not noise.*

- **`D2`** — Both 2ch cells peak at epoch 14 of 26 and both peak at exactly 2400.3 MB VRAM.
  <br/>*Why it matters:* The overfitting turn and the memory envelope are properties of the architecture + schedule, NOT of the input representation. This makes epoch count a CONFOUND for every comparison run under the fixed wall-clock budget: any change that slows an epoch silently reduces the epochs that fit, and is penalised for reasons unrelated to its mechanism.
- **`D7`** — The two I1 arms improved the composite through OPPOSITE metrics. Arm A (density only, smear unchanged): rmse -0.0227, d1 -0.0019. Arm B (6.2x finer smear): rmse -0.0093, d1 +0.0067 -- and B's RMSE is worse than A's despite B having vastly better temporal resolution.
  <br/>*Why it matters:* If temporal resolution set range accuracy, B should own RMSE. It does not; A does, and A did not change resolution at all. Meanwhile B, which also sacrifices frequency resolution (win 400 -> 64), buys ANGLE. That inverts the physical story: sharper transients seem to help azimuth cues (ILD/IPD are read across frequency and time), while range accuracy responds to something in the sampling/interpolation of the time axis.
- **`D8`** — E6 holds 29.7% LESS high-frequency azimuthal power than E2 (0.0232 vs 0.0331) yet has a BETTER d1 (0.6005 vs 0.5938). Separately, removing 58% of the gradient (E7 vs E3) barely changed the spectrum or the composite.
  <br/>*Why it matters:* It breaks the assumption -- mine, unstated until now -- that d1 improves because predictions get sharper. d1 counts pixels within +-25% of truth, and a well-centred smooth field beats a mis-placed sharp one. So the low-pass character of these models may be largely IRRELEVANT to the metric, and 'restore high frequencies' is probably the wrong research goal.

### Recent decisions

| When | Mode | Event | Note |
|---|---|---|---|
| 2026-07-10T23:38 | `verify` | experiment_completed | I19 PASSED its pre-registered falsification: every far decile improved (7-8m +0.0692, 8-9m +0.0759, 9-10m +0.0595) and the mean pr |
| 2026-07-10T22:37 | `exploit` | experiment_completed | NEW RayDPT CHAMPION 1.8962 (win5/ffn4 @ lr 3e-4, 22 epochs, stable). Closing the 2x2 OVERTURNED the previous reading: lr 3e-4 gain |
| 2026-07-10T21:33 | `exploit` | experiment_completed | Control win5/ffn4 @ lr 6e-4: composite 1.9125, stable. 2x2 complete except E23. At matched lr the fast knobs cost +0.0051 (below s |
| 2026-07-10T20:48 | `exploit` | idea_added | STRUCTURAL, not a loss change. The encoder's width axis IS time, so e3's 64 columns are depth hypotheses (d = c*t/2, 0.08-9.92m).  |
| 2026-07-10T20:33 | `exploit` | experiment_completed | fast config at lr 3e-4: composite 1.9099, stable (max mae jump 0.99), converged (best ep19/25). Matches the E11 champion (1.9093)  |
| 2026-07-10T19:33 | `exploit` | experiment_completed | I18 CONFIRMED: at lr 6e-4 the fast config is stable (mae never rose between epochs, max ratio 0.99 vs 1.46/1.51 at lr 1.2e-3) and  |
| 2026-07-10T18:30 | `exploit` | experiment_completed | DIVERGED at ep7 despite w_coarse_layout=0, never recovered (val d1 0.5342 -> 0.1307; lc pinned at 0.1849, D_coarse saturated to a  |
| 2026-07-10T18:16 | `exploit` | candidate_dropped | PREMISE REFUTED BY ITS OWN EXPERIMENT. E18 removed lc from the loss entirely and the run still destabilised (mae x1.51 at epoch 7, |

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
| 11 | `bc415a2` | 0.4187 | 1.3284 | 0.5665 | 1.9192 | keep | E10 RayDPT decode32 xlayers2 batch64 (S4: was the cross-layer cut load-bearing?) |
| 12 | `c147692` | 0.4199 | 1.3276 | 0.5710 | 1.9093 | keep | E11 RayDPT decode32 xlayers2 kv=e4 batch64 (S5/H2: first CONVERGED 2-layer RayDPT) |
| 13 | `1b995ab` | 0.4125 | 1.3409 | 0.5664 | 1.9250 | keep | E12 RayDPT decode32 xlayers1 kv=e4 (S5 attribution: 2x2 missing cell) |
| 14 | `d38ecc7` | 0.3082 | 1.5628 | 0.5464 | 2.0707 | keep | E13 RayDPT E11 arch + rel_mae dense loss (S6/I13: far-field compression) |
| 15 | `d38ecc7` | 0.4724 | 1.3606 | 0.5538 | 1.9857 | keep | E14 RayDPT E11 arch + log_mae dense loss (S6/I13 discriminating arm) |
| 16 | `789c0be` | 0.5159 | 1.3988 | 0.5137 | 2.1120 | discard | E17 DIVERGED (lc saturated ep5) FAST default: E11 arch + win32=3 + ffn=2 (F0: does the speedup cost accuracy?) |
| 17 | `e4743b7` | 0.4491 | 1.3962 | 0.5342 | 2.0424 | discard | E18 DIVERGED ep7 despite w_coarse_layout=0 -> lc is a symptom, trunk is unstable (D11 corrected) |
| 18 | `0909a2d` | 0.4201 | 1.3386 | 0.5704 | 1.9176 | keep | E20 FAST config (win32=3 ffn=2) at lr 6e-4 (F1/I18: is the instability the optimiser?) |
| 19 | `0909a2d` | 0.4270 | 1.3231 | 0.5706 | 1.9099 | keep | E21 FAST config at lr 3e-4 (F1/I18 3-trial ladder) |
| 20 | `7bf10af` | 0.4132 | 1.3357 | 0.5708 | 1.9125 | keep | E22 CONTROL win5 ffn4 @ lr 6e-4 (F1 attribution: knobs vs lr) |
| 21 | `1c34d7c` | 0.4131 | 1.3295 | 0.5765 | 1.8962 | keep | E23 CONTROL win5 ffn4 @ lr 3e-4 (F1 attribution: closes the 2x2) |
| 22 | `a7b0613` | 0.4203 | 1.3183 | 0.5733 | 1.8987 | keep | E24 EchoDelayVolume: per-ray soft-argmax over echo delay (G0/I19 structural) |
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
    subgraph MY["current — RayDPT champion (E23/E25) · 5.89M params"]
        direction TB
        A1["Binaural echo waveform (2ch)<br/>cut = 2·10m/c → 2823 samples · FIXED"] --> A2["STFT(nfft 512, hop 160, win 400)<br/>→ (F=257, T=18) → resize to 256×512<br/>height = FREQUENCY · width = TIME"]
        A2 --> A2b["named cue stack (in_ch)<br/>logL/L · logR/R · ILD · cosIPD · sinIPD"]
        A2b --> A3["UNet8Encoder, truncated at e4 (I9)<br/>e2 64×128 · e3 32×64 (freq 32, time 64) · e4 16×32"]
        A3 --> A4["RayBank ray queries Q16/Q32<br/>× audio cross-attention, 2 layers<br/>kv16 = e4 (512 tok) · kv32 = e4 (512 tok, I12)"]
        A3 --> A4s["DPT skips se3/se4<br/>added by PIXEL INDEX (I5: parked)"]
        A3 --> A7["EchoDelayVolume (I19) · optional<br/>e3's 64 TIME columns = depth hypotheses<br/>d = c·t/2 ∈ [0.08, 9.92] m"]
        A7 --> A7b["per ray: attend over FREQUENCY<br/>inside ONE time column per hypothesis<br/>→ score → softmax over DEPTH"]
        A7b --> A7c["depth = Σ p_j · d_j<br/>soft-argmax over ECHO DELAY"]
        A4 --> A5["DPT fusion at 32×64<br/>+ local spherical window attention (win 5)"]
        A4s --> A5
        A7c -->|"replaces the sigmoid coarse head<br/>that saturated and diverged (D11)"| A5
        A5 --> A6["head at 32×64 → bilinear ×8<br/>→ ERP planar depth 256×512"]
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
~58% of the gradient at convergence yet are measurably **inert** — see idea `I6`). The target is
**planar** (cubemap perpendicular-Z) ERP depth. The invariant that defines RayDPT is that depth is
decoded **per ray direction** from `RayBank` queries, never regressed from a global bottleneck.

The key structural fact, and the one `EchoDelayVolume` exploits: **the encoder's width axis is
time.** An echo from a surface at depth `d` arrives at `t = 2d/c`, so `e3`'s 64 time columns are
depth hypotheses spanning 0.08–9.92 m. Rather than making the network *learn* that correspondence,
`I19` lets each ray attend over **frequency** inside a single time column per hypothesis (azimuth
lives in the per-frequency ILD/IPD) and takes a **soft-argmax over echo delay**. Measured against
its matched control, every far decile improved (7–8 m +0.069, 8–9 m +0.076, 9–10 m +0.060) and the
mean predicted depth of a true 8.5 m surface rose from 4.76 m to 5.24 m.
