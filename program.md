# Auto Audio Depth Estimation

Project slug: `auto-audio-depth-estimation`

Autonomous research for **depth estimation from binaural echoes**. An AI agent iteratively forms a
hypothesis, modifies `train.py`, trains under a fixed budget, evaluates, concludes PASS/FAIL, records
the scientific finding, and continues indefinitely. This file is the agent's operating manual.

> **Phase note — 2026-July RayDPT-validation (FRESH RESET).** This branch (`master` /
> `2026-July-RayDPT-validation`) is a fresh restart from the **original RayDPT baseline** (`f677b0f`, 5ch)
> with a **corrected loss** (the coarse-layout & low-pass loss *targets* now use MASK-WEIGHTED pooling —
> the old code averaged/blurred `gt` over invalid `gt=0` pixels, corrupting the target; `compute_errors`
> is unchanged). Goal: **re-validate** the research under the fixed loss.
> The **June improvement phase** (champion 12ch multi-res+coherence+TTA, composite ~2.030, ABS_REL −26% /
> RMSE ~−9% / d1 +7pts vs baseline) is fully archived on branch **`2026-June-RayDPT-improvement`** (commit
> `ff22f23`, 164 runs / 29 studies) — read it for the prior findings to re-test.
> The framework is unchanged (hypothesis-driven workflow below; `utils/research.py`, `out/hypothesis.tsv`, `studies.json`). **Start from E0 (baseline) this phase.** Run `python utils/research.py status`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch_audio/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that the dataset directory exists. If not, tell the human to run `conda activate ss && python prepare.py`.
5. **Initialize out/results.tsv**: Create `out/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 1 hour** (`TIME_BUDGET = 3600` in `train.py`, wall clock training time, excluding startup/compilation). Training automatically stops when the budget is reached. You launch it simply as: `conda activate ss && python train.py --mode train`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify the **FIXED** parts of `prepare.py`: `get_scene_split` (data split), `SoundSpacesDataset._wave`
  (waveform access), `._depth` (target depth), `compute_errors` (metric), `swap_audio_lr` (L/R symmetry).
  These define the benchmark and keep E0–E134 reproducible. Do not touch them.
- Install new packages or add dependencies. You can only use what's already in the conda `ss` env.
- Modify the evaluation harness. The `compute_errors` function in `prepare.py` is the ground truth metric.

**What you CAN now do (PROPOSAL-01, implemented):** research the **acoustic representation** — the
waveform→input-feature mapping — via the `prepare.FEATURE_FN` seam, set from `train.py`. It receives the
FIXED raw waveform and the dataset instance and returns `(in_ch, H, W)`. When `FEATURE_FN is None` the
pipeline is byte-identical to the 5ch baseline (verified). This unblocks multi-resolution STFT, early/late
echo split, cross-channel coherence, etc. Keep the default representation as the baseline; never change the
fixed split/target/metric.

**Training objective (RayDPT).** The model is trained with the composite loss in `composite_loss` (`train.py`):

```
loss = w_dense·main  +  w_coarse_layout·lc  +  w_low·llow
```

- `main` — dense masked-MAE on the full-resolution depth `D`. This is the **only required** term (`w_dense=1.0`).
- `lc` — coarse 16×32 layout MAE on `D_coarse` (`w_coarse_layout=1.0`). **Not mandatory** — a layout regularizer.
- `llow` — low-pass MAE on `gaussian_blur_erp(D)` (`w_low=0.5`). **Not mandatory** — a smoothness / low-frequency regularizer.

`lc` and `llow` are auxiliary knobs: set their weights to `0` (or drop them) to train on the dense term alone. They bias the model toward correct global layout but are not required for training to run, and are fair game to tune/disable when chasing ABS_REL/RMSE. (Note: `prepare.py._depth` also converts the dataset's cubemap perpendicular-Z depth to **radial** depth — RayDPT trains and is evaluated on radial depth.)

**Default config = `C_raydpt_5chflip`, but these are knobs, not requirements.** The first run establishes the baseline with the defaults: `--in-ch 5` (RIR spatial feature `[logL,logR,ILD,cosIPD,sinIPD]`) + `--flip-aug True` (L/R mirror augmentation) + radial depth. None of these are mandatory — `--in-ch {2,3,5}`, `--flip-aug {True,False}`, the loss weights, optimizer, LR, batch size, model width/depth, number of attention layers, etc. are all fair game to lower ABS_REL/RMSE.

**The one architectural invariant: keep the model RAY-CONDITIONED.** RayDPT's essence is per-ray spherical queries — the fixed `RayBank` on the ERP ray grid — that cross-attend the audio tokens, so depth is decoded *per ray direction*, not as a plain pixel map regressed from a global bottleneck. You may restructure almost anything else, but **do not remove the ray-conditioning** (RayBank ray queries × audio cross-attention): that is the hypothesis under test. A pure encoder→pixel-decoder with no ray queries is out of scope.

**The goal is simple: get the lowest ABS_REL and RMSE errors** (judged via the honest composite — see below). The time budget is fixed at **1 hour** (`TIME_BUDGET = 3600`), so training time per run is effectively constant; heavier configs simply fit fewer anneal epochs. (Note: an earlier revision of this file said "30 minutes" — that was stale; the budget is 1 hour, matching `TIME_BUDGET` and the "1 hour" statement above.) Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful ABS_REL and RMSE gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 ABS_REL/RMSE improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 ABS_REL/RMSE improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints validation metrics each epoch like this:

```
Epoch [5/40] Loss: 0.1328 Time: 156.1s
  Val Loss: 0.1430 | ABS_REL: 0.5690 RMSE: 1.1513 d1: 0.4687 d2: 0.6871 d3: 0.8139
  >> Best model saved (ABS_REL: 0.5690)
```

And at the end of training:

```
Training complete. Best ABS_REL: 0.5690
```

You can extract the key metrics from the log file:

```
grep "ABS_REL\|RMSE" run.log
```

## Logging results

When an experiment is done, log it to **`out/results.tsv`** (tab-separated). The TSV has a header row and 7 columns:

```
commit	abs_rel	rmse	d1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. ABS_REL achieved (e.g. 0.5889) — use 0.0000 for crashes
3. RMSE achieved (e.g. 1.1503) — use 0.0000 for crashes
4. d1 (delta < 1.25) accuracy (e.g. 0.4782) — use 0.0000 for crashes
5. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. **SHORT description (≤ ~120 chars): `Exx [Sxx type] <what changed>: <comp> vs champ <val> -> keep/discard`.**
   Keep it terse — one line, the change + the verdict number. Put any long reasoning/mechanism in the
   study record (`out/hypothesis_details.tsv`), NOT here.

Example:

```
commit	abs_rel	rmse	d1	memory_gb	status	description
a1b2c3d	0.5889	1.1503	0.4782	12.3	keep	baseline
b2c3d4e	0.5200	1.0800	0.5100	12.5	keep	increase LR to 0.001
c3d4e5f	0.6100	1.2000	0.4500	12.3	discard	switch to GeLU activation
d4e5f6g	0.0000	0.0000	0.0000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch_audio/mar5` or `autoresearch_audio/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `conda activate ss && python train.py --mode train > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "ABS_REL\|RMSE\|Best" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in `out/results.tsv`
8. If ABS_REL improved (lower), you "advance" the branch, keeping the git commit
9. If ABS_REL is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~60 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 70 minutes, kill it and treat it as a failure (discard and revert). ###

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~60 minutes then you can run approx 1/hour, for a total of about 8.5 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Autonomous continuous operation (standing directive)

Run **fully autonomously and indefinitely**. The moment one run finishes, record it and launch the next — never pause, never ask. Between launches the single GPU is busy ~1 hr/run; schedule a wakeup to poll and resume the loop so it self-continues across turns. Always have the next experiment staged.

**Devise your own improvements** — you have full latitude over `train.py`: model layers, the ray↔audio **cross-attention** and ray↔ray **self/local attention** depth & width, attention heads, the loss terms/weights, optimizer, schedule, regularization, augmentation, etc. When ideas run low, re-read the in-scope files and the source repo (`test_for_audio_implicit_full`), combine prior near-misses, and try more radical restructurings — while keeping the ray-conditioning invariant.

**Judge holistically (ABS_REL + RMSE + d1 together).** ABS_REL is directly optimizable by a relative loss, so it can be "gamed"; treat **RMSE and d1** as the honest quality signals and do not crown a config that only wins ABS_REL while RMSE/d1 regress. Select the best epoch/checkpoint by a multi-metric composite, not ABS_REL alone.

**Respect the fixed 1-hour budget.** Heavy capacity (full-decode head, deeper cross-attn, more heads) slows epochs → fewer anneal steps → busts the budget and loses on RMSE/d1. Favor light/fast configs that fit ~7 epochs; prefer levers that don't slow training.

**Keep `out/results.tsv` and the hypothesis TSVs current** and `git push` after each experiment.

---

# Research workflow (v2) — hypothesis-driven autonomous research

The keep/discard loop above still runs each experiment. This section adds lightweight *scientific
structure* around it so research explores real mechanisms instead of collapsing into endless
hyperparameter tuning. Keep it simple — it is four small files plus `research.py`, not a framework.

## Repo layout & state files (all human-readable; edit by hand)
```
train.py        prepare.py(FIXED)   program.md   README.md(status display)   studies.json
utils/research.py            out/results.tsv  out/hypothesis.tsv  out/hypothesis_details.tsv
```
- **`out/results.tsv`** — authoritative per-run log, one SHORT row per run (see "Logging results").
- **`out/hypothesis.tsv`** — one SHORT row per *study* (mechanism, not a single run): `study_id, lineage,
  type, conclusion (PASS/FAIL/NEUTRAL), best_comp, best_exp_id, best_commit, one-line summary`.
- **`out/hypothesis_details.tsv`** — the LONG per-study content (general + detailed hypothesis, experiment
  note, exp IDs, HPO count, scientific conclusion, failure mode), keyed by `study_id`. Keep the verbose
  reasoning HERE so results.tsv / hypothesis.tsv stay terse.
- **`studies.json`** — the *active* study, its adaptive-HPO stage, `next_exp_id`, the current
  `global_champion` (mechanism-keyed), and the backlog.
- **`utils/research.py`** — `python utils/research.py status` prints the state; `... composite --abs_rel A
  --rmse R --d1 D` computes the honest composite; `... next-id` prints the next experiment id.
(No `archive.json` / `EXPERIMENTS.md` — the champion lives in `studies.json`; per-study findings live in the
two hypothesis TSVs.)

## Three-level hypothesis (every NEW/refine/combine study must state all three)
1. **General hypothesis** — WHY this direction matters (a problem, limitation, or principle). Not a
   parameter value. *Good:* "Global spatial reasoning may be more effective after local acoustic
   evidence has been fused into a coherent coarse scene." *Bad:* "Add one attention block."
2. **Detailed hypothesis** — HOW the proposed mechanism is expected to work and why it may help.
3. **Experiment note** — WHAT actually changed in this run (the code/param diff).

Do not confuse the levels. A module insertion or LR value is an experiment note, never a general
hypothesis.

## Experiment types (exactly one per experiment)
- **new** — a qualitatively new mechanism / research hypothesis (not just a new parameter value).
- **refine** — improve the implementation/formulation of an existing mechanism (better placement,
  more faithful formulation, fix a weakness) without changing its central hypothesis.
- **tune** — change hyperparameters while preserving the method (LR, loss weights, depth, heads,
  schedule, regularization strength…). HPO is allowed but must *serve* a structural idea.
- **combine** — merge compatible mechanisms from distinct lineages. State *why* they are
  complementary first; do not combine merely because both scored well.
- **confirm** — repeat/validate a config when the gain is near noise, reproducibility is uncertain,
  or a major conclusion rests on one run. A confirm must reference the study/exp/config it confirms.

## Adaptive HPO ladder (tuning serves research; do not endlessly tune a weak idea)
Run the structural idea FIRST (Stage 0 screen). If it is catastrophically worse, broken in
principle, or cannot support its own hypothesis → conclude **FAIL** without mandatory HPO. If it is
plausible / competitive / ambiguous / promising / a useful specialist → begin HPO:
`structural-screen → 3 → 5 → 7 → 10`, each extension justified by accumulated evidence.
- **3** is the minimum HPO budget for a viable idea (spend on the most consequential params).
- extend to **5** if it improved meaningfully / approaches its parent / looks under-tuned / shows a
  useful metric tradeoff.
- extend to **7** if it beats or closely competes with its parent lineage.
- **10** is exceptional — only if it is a serious final-method candidate still improving.
Do not jump straight from one structural run to ten tuning runs.

## Decision policy (prospective only — never rewrite past decisions)
Judge on the **honest composite** `rmse/1.6 + (1-d1)/0.46 + 0.35·abs_rel` (lower better), with
RMSE + d1 dominant and ABS_REL discounted (it is directly optimised → gameable). Also weigh:
individual-metric regressions, Pareto behaviour, estimated noise, model complexity, VRAM/compute,
and interpretability. Never let a method win one metric while badly regressing the others.

## Noise & confirmation
Noise σ ≈ **0.008** (20+ reruns), up to **~0.014–0.019** on small samples. Rules:
`large clear improvement → provisional PASS candidate`; `near-noise improvement → confirm required`;
`consistent confirmation → strengthen`; `inconsistent → mark uncertain / one more justified confirm`.
**Never crown a sub-0.015 candidate on fewer than 3 confirming draws** — see study **S10** (E121–E125:
crowned on two lucky low draws, demoted when the 3rd draw exposed it as noise; and S04/E60).

## Multi-lineage archive
The repo is no longer a single champion trajectory. Maintain lineages (mechanism families) with their
own champions in `studies.json`. A method may FAIL to become global champion yet PASS as a hypothesis
if it shows a clear, reproducible effect, a useful specialist profile, or a meaningful tradeoff worth
recombining later. A tiny metric gain is not automatically a strong result.

## After every study — pick ONE next action, then continue
`explore` (new hypothesis; on plateau/diminishing returns/new failure mode) · `refine` (valid idea,
incomplete implementation/wrong location) · `tune` (viable mechanism, unmapped param range) ·
`combine` (two lineages address different failure modes, compatible, interaction explainable) ·
`confirm` (near noise / conflicting / champion candidate / conclusion rests on one run).
Then append the study conclusion to `out/hypothesis.tsv` (+ `out/hypothesis_details.tsv`), update
`global_champion`/`next_exp_id` in `studies.json`, `git push`, and launch the next run.

## Continue indefinitely
Do not stop after a failed run, an eval error, a failed hypothesis, a completed HPO stage, a finished
study, a PASS, a FAIL, a confirmation, or a lineage plateau. On error: log the failure record, recover
to a valid parent / lineage champion, diagnose briefly, continue. On plateau: reduce local tuning and
explore a different mechanism or revisit an informative abandoned branch. No background daemon — just
keep following this file. The loop runs until externally stopped.

---

# PROPOSAL-01 — acoustic-representation refactor (IMPLEMENTED, operator-approved)

**Status: DONE.** The `prepare.FEATURE_FN` seam is implemented and baseline byte-equivalence is verified
(FEATURE_FN=None reproduces the 5ch representation exactly). Fixed split/target/metric/waveform are
untouched. Acoustic-representation research is now an open lineage — set `prepare.FEATURE_FN` from
`train.py` and set the model `in_ch` to match the produced channel count. Original proposal below.

---


**Observation.** The audio representation (STFT + binaural spatial cues `[logL,logR,ILD,cosIPD,sinIPD]`)
is constructed inside `prepare.py` (`SoundSpacesDataset._specN` / `_spec2`, with fixed `_NFFT/_WIN/_HOP`),
which is read-only. Study **S07** (E70/E74/E80/E81/E108/E109) could therefore only test coarse choices
(in_ch, augmentation, positional info) and found the current 5ch cues load-bearing. Deeper acoustic
representation research (multi-resolution STFT, early/late echo split, cross-channel coherence,
frequency-dependent cue processing) is currently **blocked** by the fixed pipeline.

**Proposal (smallest safe refactor).** Separate concerns so representation logic becomes editable
research while reproducibility-critical parts stay fixed and unchanged:
- **Fixed (never edit — preserves benchmark & old scores):** dataset split (`get_scene_split`), sample
  identity, target radial depth (`_depth`), evaluation (`compute_errors`), and the L/R symmetry helper
  (`swap_audio_lr`).
- **Editable research logic (move behind a clear seam):** raw-waveform→feature construction
  (`_specN`/`_spec2`, STFT params, spatial-cue math) — ideally exposed as a function `train.py` can
  override or select, with the current 5ch construction as the **default baseline**.

**Risk / status.** This changes `prepare.py`, which is currently declared read-only, so it must be done
carefully: keep the current representation as the exact default, verify byte-identical features for the
baseline (so E0–E131 remain reproducible), and never alter the split/target/metric. **Not implemented
now** — recorded here as an infrastructure proposal. Only implement when an acoustic-representation
study is the chosen next action, and gate it behind an explicit baseline-equivalence check.
