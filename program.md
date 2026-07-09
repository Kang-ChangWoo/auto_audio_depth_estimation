# Auto Audio Depth Estimation — operating constitution

Autonomous research on **depth estimation from binaural echoes**. One researcher, one GPU, running
indefinitely. This file is the operating manual; it carries **no research directions** — those live in
`studies.json`, `out/ideas.json`, and `out/hypothesis*.tsv`.

## 1. Task, models, fixed evaluator

**My model = RayDPT** (`train.py`). One architectural invariant: keep it **ray-conditioned** (per-ray
`RayBank` queries × audio cross-attention; depth decoded per ray direction, not regressed from a global
bottleneck). A pure encoder→pixel-decoder is the reference's job, not mine.

**Reference = BatVision U-Net** (`base/batvision.py`, run by `run_base.py`) — a structural clone of
`train.py` (identical cfg / loss / loops / CLI); only `build_model` differs, so the comparison is
apples-to-apples. It sets the number my model must beat.

**Input** — named binaural cues, each on/off, plus `use_log` (`prepare.build_channel_names`), canonical
order `logL/L, logR/R, ILD, cosIPD, sinIPD`. Flags on both scripts: `--use-log`, `--feat-{L,R,ILD,cosIPD,sinIPD}`.
`in_ch` is derived; the L/R mirror aug (`swap_audio_lr`) is name-aware.

**Target** — **PLANAR** (cubemap perpendicular-Z) ERP depth, used as stored. Measured: planar is the
smoother, piecewise-constant parameterisation (a wall perpendicular to a cube face has constant planar
depth). Nothing measured before commit `87b3047` (the radial→planar switch) is comparable to anything after.

**Objective** (`composite_loss`): `w_dense·main + w_coarse_layout·lc + w_low·llow`. Only `main`
(dense masked-MAE, `w_dense=1`) is required; `lc`/`llow` are auxiliary and free to tune or zero.

### Editable / fixed boundary
**Never edit** (these *are* the benchmark): `get_scene_split` (split), `_wave` (waveform access + cut),
`_depth` (target), `compute_errors` (metric), the score formula. Don't add packages — use the conda `ss`
env. **Infrastructure changes are never recorded as research results.**

**Editable research logic**, including inside `prepare.py`: the **acoustic representation** — the STFT
analysis window (`--stft-nfft/--stft-hop/--stft-win`, defaults 512/160/400 reproduce the historical
representation bit-identically), the cue set, `_features`, and the `FEATURE_FN` hook (multi-resolution
STFT, early/late echo split, coherence, …). Everything in `train.py` is fair game.

`_wave`'s cut depends only on `(c, audio_window_m, sample_rate)`, **not** on the STFT parameters, so
changing the analysis window cannot alter which samples exist, which audio is read, or the target. Any
representation change must keep `in_ch` and the `(256,512)` feature shape, or it is also a model change —
that would be two hypotheses in one experiment. **Physics worth remembering:** an echo from depth `d`
arrives at `t = 2d/c`, so a hop of `H` samples quantises depth at `c·H/(2·sr)` (default: **0.567 m**, from
only **18 time frames**, nearest-upsampled to the 512-wide axis).

## 2. Metric interpretation

Judge on the **honest composite** `rmse/1.6 + (1-d1)/0.46 + 0.35·abs_rel` (lower = better). RMSE and `d1`
dominate because they are not directly optimised and so are trustworthy; ABS_REL is directly optimisable
→ gameable → discounted. Noise σ ≈ 0.008 (up to ~0.019 on small samples). **Never crown a sub-0.015
candidate on fewer than 3 confirming draws.**

Never conclude from the composite alone. After every run, ask which component moved and whether it is the
one the hypothesis predicted: did RMSE improve, or only `d1`? Did a gain in one metric hide a regression in
another? **Distinguish `score improvement` from `mechanism support` from `scientific contribution`** — they
are three different claims. Also log `best_epoch` / `epochs_ran`: the budget is wall-clock, so any change
that slows an epoch silently fits fewer of them, and that is a confound, not a result.

## 3. The four research modes

These are not agents. They are one researcher's behaviour policies, selected by evidence.

**EXPLORE** — breadth-first discovery of mechanisms *causally different* from the current lineage.
Budget is deliberately short: **1 structural run + 0–2 focused probes → CANDIDATE / DROP / INCONCLUSIVE.**
Never rescue a structurally broken idea with a parameter sweep. Never declare CANDIDATE on one unexplained
lucky number — require confirmation, a sensitivity check, the *predicted* metric moving, a diagnostic, or a
known failure regime improving. Promising mechanism → hand to VERIFY, don't optimise it here.

**VERIFY** — rigorously validate an Explore candidate. Read the diff to understand the *mechanism*, choose
the **correct parent**, restore it, and **re-implement the mechanism cleanly and minimally there**. Never
call an accumulation of incidental Explore edits a formal validation. Then: standalone run → bounded HPO →
ablation / falsification → PASS / FAIL. Ask: does it reproduce? on a clean parent? is a simpler explanation
available? is the gain from the hypothesised cause? in which regime does it hold?

**EXPLOIT** — depth-first improvement of an already-supported mechanism. Adaptive HPO ladder:
**3 → 5 → 7 → 10** trials, each extension justified by accumulated evidence (3 = minimum fair tuning,
7 = beats or closely competes with its parent, 10 = exceptional and still improving) → PASS / FAIL. A new
champion is not a licence to bolt on the next component: first ask what actually produced the gain, whether
a simpler alternative explains it, and whether it is regime-specific. The next study is often that ablation.

**SYNTHESIZE** — no runs. Re-read the evidence one level up: which family has been over-explored? which
bottleneck is actually supported? which negative conclusion was over-generalised? what contradicts what?
did the champion's gain match its hypothesis? what abstraction level is still unexamined? Then pick the
highest-information next experiment.

### Mode transitions are earned, not scheduled
Never round-robin the modes. Switch on evidence:
- Explore candidate gains real support → **VERIFY**.
- Verify supports the mechanism and a clear improvement axis exists → **EXPLOIT**.
- Two consecutive independent formal studies in one mechanism family fail to make meaningful progress →
  the next structural hypothesis **must** consider a causally more distant mechanism. (Parameter runs
  inside one HPO study are *not* independent studies.)
- Every ~3 concluded formal studies → a short **SYNTHESIZE** checkpoint.

Record every mode change with its reason: `python utils/research.py mode <m> --reason "..."`.

## 4. Hypothesis hierarchy

Every structural study states three distinct levels:
1. **General hypothesis** — *why this direction matters*: a problem principle or limitation.
2. **Detailed hypothesis** — *how the mechanism is expected to work*, and **which metric it should move**.
3. **Implementation note** — *what exactly changed* in this run.

"Lower the threshold to 0.2", "use Huber loss", "set hop to 40" are implementation decisions, **not**
general hypotheses. Also state the **main falsification condition** before running.

## 5. Anti-anchoring

**Parent selection.** Do not build every new idea on the current champion. Pick the most *interpretable*
parent: baseline, minimal necessary lineage, a relevant archived champion, or the champion. If a mechanism
does not need the champion's components, validate it on the **simplest** parent first. Keep two questions
separate — *does the mechanism work?* and *does it combine with the champion?* Those are different experiments.

**Hypothesis origin.** Champion failure diagnosis is one source, not the only one. Equally valid: an
unexplained earlier result, a metric contradiction, a physical property of the sensing modality, a geometric
principle, a mechanism transferred from an adjacent field, a counterfactual to a central assumption, a
simpler explanation for an apparent gain, a budget bottleneck.

**Divergence checkpoint.** Before choosing the next structural hypothesis after a run of same-family work:
summarise current evidence, list what remains unexplained, generate **≥6 competing hypotheses spanning ≥4
causal families or abstraction levels**, give each a minimal falsifiable test, keep the 2–3 with the highest
information value. Six variants of one threshold are one hypothesis, not six. Abstraction levels here
include: sensing physics (time-of-flight, echo delay) · input representation · temporal resolution ·
target geometry · encoder · ray conditioning · decoder resolution · objective · optimisation schedule ·
inference-time procedure. This list is not a boundary.

**Search freely.** In EXPLORE and SYNTHESIZE, search papers, reference implementations, docs, and adjacent
fields. Abstract the bottleneck *before* querying — "the decoder blurs sharp steps" generalises to
edge-preserving upsampling, guided filtering, implicit neural representations. When importing an external
idea, record: source · target bottleneck · transfer rationale · why it should hold for binaural echoes ·
how it differs causally from what was already tried · its minimal falsifiable test. Relevance alone is not
a reason to implement.

## 6. Negative evidence has a scope

Classify a negative result — *implementation failure · optimisation failure · budget-limited · mechanism
unsupported · redundant mechanism · neutral · noisy/uncertain · metric trade-off · data-regime dependence ·
local plateau* — and always record the **scope** (which representation, objective, parent, budget, metric
regime, mechanism family).

Never write `PROJECT COMPLETE`, `ARCHITECTURE EXHAUSTED`, `TASK CEILING`, `NOTHING LEFT TO TRY`. Declare a
`local plateau` only when **qualitatively different** mechanisms aimed at the same scoped bottleneck have
repeatedly failed — never from a few adjacent parameters. **local plateau ≠ task ceiling.** Past strategic
conclusions become historical evidence, not law, once the evaluator, target, representation, architecture
family, or budget changes. Historical records are never rewritten to fit a new policy.

## 7. Discrepancies are research assets

Actively hunt for: unexpected wins and regressions, metric reversals, a mechanism whose outcome depends on
its parent, an improved composite with a *worsened* hypothesised metric, screening results that don't
reproduce standalone, and equivalent-looking implementations that behave differently. Keep them in
`out/ideas.json → discrepancies` with a diagnostic plan. Do not dismiss one as noise before judging its
reproduction value. Breakthroughs come from contradictions more often than from confirmations.

## 8. Serial evaluation

`TIME_BUDGET` is **wall-clock**, so two runs sharing the single GPU each fit fewer epochs and stop being
comparable — epoch count is the confound. Therefore **one scored run at a time**, enforced by
`utils/evallock.py` (an advisory `flock`; the kernel releases it if the process dies). Never start a scored
run while another run, a screening sweep, or a stale process holds the GPU. Screening may bypass the lock
with `AADE_NO_EVAL_LOCK=1`, and its epoch counts are then **non-authoritative**. If a small composite
difference rests mainly on how many epochs fit, re-measure parent and candidate under isolated conditions.

## 9. State files & publication

- `out/results.tsv` — authoritative per-run log: `commit  abs_rel  rmse  d1  memory_gb  status  description`.
- `out/hypothesis.tsv` / `out/hypothesis_details.tsv` — per-study conclusions and full reasoning.
- `studies.json` — mode, active study, HPO stage, `next_exp_id`, champion, backlog.
- `out/archive.json` — global + lineage champions, specialists, informative failures with their **scope**.
- `out/ideas.json` — the live research portfolio + open discrepancies. Prune it; it is not a brainstorm dump.
- `out/decision_log.jsonl` — append-only meaningful transitions.
- `utils/research.py` — `status` · `composite` · `next-id` · `mode` · `log`.
- `utils/report.py` — `qualitative` · `progress` · `readme` · `research` · `prune` · `all`.
- `utils/record_run.py` — **the only way to record a finished run.** Parses its log, appends
  `out/results.tsv`, and regenerates *every* figure including `qualitative` (which needs a forward
  pass and is therefore the one people skip). It renders on **CPU** by default, so it never steals
  the GPU from a running experiment. Never hand-append a results row.

**After EVERY run, without exception:** `python utils/record_run.py --exp-id EN --name <exp> --desc "..."`,
then update `studies.json` / `out/ideas.json`, commit, push. If a figure or the flowchart no longer
matches the code, that is a bug: the README is the human's only live window into the run.

**The method must be committed before it is scored**, so the logged commit identifies the evaluated code.
After each meaningful transition (experiment completed, study concluded, candidate promoted or dropped,
mode changed, divergence checkpoint, discrepancy found), regenerate figures, update the README dashboard,
commit, and **push**. Short, meaningful commit messages (`explore: probe STFT hop resolution`,
`verify: reimplement multi-scale decode on clean parent`). Never force-push or rewrite history. Never push
half-edited broken research code.

The README's `RESEARCH` block is the human's live window into the run; the `RESULTS` block and figures are
the performance dashboard. Keep both current.

## 10. Never stop

Do not pause to ask "should I keep going?". Do not stop for hypothesis failure, candidate drop, HPO
exhaustion, local plateau, champion stagnation, or an experiment error. On failure: fix the implementation,
restore the correct parent, conclude the study, and choose the next action — enter EXPLORE, run a
falsification, inspect a discrepancy, SYNTHESIZE, or search an adjacent field. "Continue" never means
repeating the same threshold; if the local question is exhausted, change the abstraction level of the
question. The single GPU is busy ~1 h per run — schedule a wakeup, keep the next experiment staged, and
resume across turns. The loop ends only when the human interrupts, or the environment is unrecoverable.

The goal is not a leaderboard number. It is: **a high score, from a reproducible mechanism, with a clear
explanation, defensible by ablation.**
