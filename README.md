# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP planar (cubemap) depth (SoundSpaces).

<!-- RESEARCH:START -->
## Autonomous research state

| | |
|---|---|
| **Mode** | `EXPLOIT` — adaptive HPO ladder 3 -> 5 -> 7 -> 10, each step justified by evidence -> PASS / FAIL |
| **Active study** | `F0` [refine] raydpt-fast-baseline (*running*) |
| **Research question** | Research throughput is a first-class research variable. Under a fixed wall-clock budget, a cheaper model that converges further is not merely faster to study -- it scores better, and it lets more hypo |
| **Current action** | E17 raydpt_e17_fastbase: the new defaults, scored at the full 3600s budget, --epochs 28. |
| **Latest result** | *(no scored run in this study yet)* |
| **Next decision** | Adopt iff E17's composite is within sigma (0.008) of E11's 1.9093, or better. Both knobs are CAPACITY cuts, so speed alone proves nothing. A win does not prove the knobs are free -- it proves the trad |
| **Why this mode** | Operator asked for a good starting RayDPT that is fast to experiment with. The bottleneck on iteration speed is measured, not guessed: cr32 37.1% and lsa32 33.5% of forward. |

### Current hypothesis

- **General** — Research throughput is a first-class research variable. Under a fixed wall-clock budget, a cheaper model that converges further is not merely faster to study -- it scores better, and it lets more hypotheses be tested per day. But a speed change that quietly costs accuracy buys nothing, so every knob must be paid for in a scored run.
- **Detailed** — GPU profiling of the E11 champion (batch 64, bf16): cr32 37.1%, lsa32 33.5%, enc 13.9%, cr16 10.0% of forward. Two knobs follow. raydpt_win32 5->3 shrinks the local spherical attention from 25 offsets to 9 (measured 1.20x). ffn_mult 4->2 halves the CrossBlock FFN (1.10x on top). Together 169.1 -> 127.7 s/epoch, 21.3 -> 28.2 epochs/h, 5.89M -> 5.29M params.
- **Implementation note** — E17 raydpt_e17_fastbase: the new defaults, scored at the full 3600s budget, --epochs 28.

### Research portfolio

| Idea | Mechanism family | Causal distance | Target bottleneck | Status | Next test |
|---|---|---|---|---|---|
| `I1` | acoustic-representation / temporal resolution | far | time-of-flight quantisation in the input representation | inconclusive | HOLD. Do not run more I1 arms. I10 (bilinear at hop=160) isolates the staircase; interpret |
| `I3` | training-optimization | near | the 1h wall-clock budget is spent on epochs that make the model worse | backlog | queue after the RayDPT planar re-anchor (E4); this is a confound affecting EVERY future ru |
| `I5` | ray conditioning / encoder-decoder correspondence | mid | RayDPT's DPT skip connections impose a FALSE spatial correspondence between the spectrogram's axes and the ERP's axes | inconclusive | none. Do not spend GPU on the skip ablation on this rationale. Revive only with an indepen |
| `I7` | sensing physics / angular resolution | far | two microphones may fundamentally under-determine high azimuthal frequencies | candidate | Do not chase high-frequency power as a goal. Re-test the observability claim once RayDPT c |
| `I10` | acoustic-representation / interpolation | mid | the nearest-neighbour resize in _features() turns the time axis into a coarse staircase | inconclusive | deferred confirm: run `--feat-interp bilinear --stft-hop 40` after the RayDPT throughput s |
| `I14` | ray conditioning / audio token routing | mid | far-field rays cannot see the late, weak echo that carries distance | probing | E16 (control) then E15b (treatment), both at lr 6e-4. Pre-registered falsification unchang |
| `I15` | architecture / stability | near | RayDPT's coarse head is a sigmoid one 1x1 conv away from the deepest ray-attention output, with no normalisation between | backlog | deferred until I14 is decided at the safe lr; a stability fix must not be bundled with the |
| `I16` | throughput / experiment economics | near | iteration speed IS a research variable under a wall-clock budget | inconclusive | Re-test the fast knobs only after training is stable (E18/E19). Until then the fast defaul |
| `I17` | simplification / stability | near | RayDPT's coarse-layout auxiliary is its only failure point, and it was already measured inert | probing | E18 running; E19 (weight 0.1) follows as a second stability data point. |

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
| 2026-07-10T17:58 | `exploit` | experiment_completed | DIVERGED at epoch 5 (lc 0.0633 -> 0.3209, never recovers). Recorded as discard. Not a test of the fast knobs' accuracy -- a second |
| 2026-07-10T16:44 | `explore` | discrepancy_recorded | D11: E15 DIVERGED at epoch 4 with the parent's lr. The blow-up is entirely in lc, the coarse-layout term (0.4402 vs 0.0618) -- and |
| 2026-07-10T16:25 | `explore` | idea_added | From data already collected: E9 (cr32 on 2048 fine tokens) scores d1 0.2544 at GT 8-9m; E12 (512 coarse) only 0.1579. Far surfaces |
| 2026-07-10T16:25 | `explore` | candidate_dropped | PRE-REGISTERED FALSIFICATION MET: both relative dense terms made the 7-10m deciles WORSE (rel_mae -0.19/-0.13/-0.22; log_mae -0.13 |
| 2026-07-10T16:25 | `explore` | experiment_completed | log_mae also fails: d1 0.5538 (-0.0172), rmse +0.0330. Symmetric in the ratio, so relativity itself -- not rel_mae's asymmetry --  |
| 2026-07-10T15:30 | `explore` | experiment_completed | rel_mae FAILED: d1 -0.0246, rmse +0.2352, composite 1.9093 -> 2.0707. My sign error: \|D-gt\|/gt makes far errors CHEAPER, and it  |
| 2026-07-10T14:17 | `explore` | direction_changed | D9 REFRAMED by full-val spatial decomposition. The deficit is NOT angular: flat across azimuth (std 0.0056), ZERO on the floor, ex |
| 2026-07-10T14:17 | `explore` | experiment_completed | 2x2 complete. Attribution: coarse KV +0.0033 d1, second cross layer +0.0046; additive, neither alone clears sigma. |

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
| 16 | `789c0be` | 0.5159 | 1.3988 | 0.5137 | 2.1120 | keep | E17 FAST default: E11 arch + win32=3 + ffn=2 (F0: does the speedup cost accuracy?) |
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
