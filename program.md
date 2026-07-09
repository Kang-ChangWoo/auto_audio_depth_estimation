# Auto Audio Depth Estimation

Project slug: `auto-audio-depth-estimation`

Autonomous research for **depth estimation from binaural echoes**. The agent forms a hypothesis,
edits `train.py`, trains under a fixed budget, evaluates, concludes PASS/FAIL, records the finding,
and continues — indefinitely. This file is the **operating manual**: how we run experiments and what
is fixed. It intentionally carries **no research directions** (what mechanism to try next) — those
live in the study records (`out/hypothesis*.tsv`, `studies.json`), not here.

## State & branches
- Experiments run on the active phase branch (currently `2026-July-RayDPT-validation` / `master`).
  Each run gets one commit; `git push` after each.
- The June improvement phase is archived on `2026-June-RayDPT-improvement` (read-only history).

## Models, inputs & objective

**My model = RayDPT** (`train.py`) — the ray-conditioned model you edit. One architectural invariant:
keep it **RAY-CONDITIONED** (per-ray `RayBank` queries × audio cross-attention; depth decoded per ray
direction, not regressed from a global bottleneck). A pure encoder→pixel-decoder is out of scope — that
is the reference's job.

**Reference model = BatVision U-Net** (`base/batvision.py`) — the plain pix2pix/CycleGAN `unet_256`
(model only) from [AmandineBtto/Batvision-Dataset](https://github.com/AmandineBtto/Batvision-Dataset).
`run_base.py` is a **structural clone** of `train.py` (identical cfg / composite loss / train-eval-CLI /
fixed `prepare.py` harness; only `build_model` differs) so the comparison is apples-to-apples. It sets the
number my model must beat. Run: `conda activate ss && python run_base.py --mode train`; add
`--epochs 1 --max-iters 1 --max-val-batches N` for a 1-iteration pipeline/visual smoke.

**Input representation** — named binaural cues, each on/off, plus a `use_log` switch
(`prepare.build_channel_names` / `SoundSpacesDataset._features`), canonical order
`logL/L, logR/R, ILD, cosIPD, sinIPD`. Flags on both scripts: `--use-log`, `--feat-L`, `--feat-R`,
`--feat-ILD`, `--feat-cosIPD`, `--feat-sinIPD` (all `True/False`). `in_ch` is derived from the enabled
cues and the L/R mirror aug (`swap_audio_lr`) is name-aware. Default (all on, `use_log=True`) = the 5ch
`[logL,logR,ILD,cosIPD,sinIPD]` stack.

**Objective** (`composite_loss`): `loss = w_dense·main + w_coarse_layout·lc + w_low·llow`.
`main` = dense masked-MAE on full-res depth `D` (the **only required** term, `w_dense=1`). `lc` = coarse
16×32 layout MAE, `llow` = low-pass MAE — auxiliary regularizers, free to tune or zero. Depth is radial
(`prepare._depth` converts cubemap perpendicular-Z → along-ray depth).

**Fixed & off-limits** (define the benchmark; never edit): `get_scene_split` (split), `_wave`
(waveform), `_depth` (target), `compute_errors` (metric). Don't add packages — use the conda `ss` env.
Everything else in `train.py` (architecture, optimizer, schedule, batch/model size, loss weights,
augmentation, the representation toggles) is fair game.

## Setup (new run)
1. Confirm the active branch — a run gets one commit.
2. Read the in-scope files: `README.md`; `prepare.py` (fixed split/waveform/target/metric — do not
   touch; the representation is editable); `train.py` (the file you edit); `run_base.py` + `base/`
   (the reference).
3. Verify the dataset dir exists (else `conda activate ss && python prepare.py`).
4. Ensure `out/results.tsv` has its header row. The first run establishes the baseline.

## The experiment loop
Single GPU, fixed **1-hour** budget (`TIME_BUDGET = 3600`, wall-clock training; heavier configs simply
fit fewer anneal epochs). LOOP:
1. Edit `train.py` with **one** hypothesis.
2. `git commit`.
3. `conda activate ss && python train.py --mode train > run.log 2>&1` (redirect — do not tee/flood context).
4. `grep "ABS_REL\|RMSE\|Best" run.log`. Empty ⇒ crashed: `tail -n 50 run.log`, fix if trivial and re-run,
   else log `crash` and move on.
5. Record to `out/results.tsv`; regenerate figures (`python utils/report.py all`).
6. **Keep** (advance the commit) if it improves the honest composite; else **revert**. Rewind sparingly.

Timeout: kill any run past ~70 min and treat it as a failure. VRAM is a soft constraint (moderate
increases OK for real gains, no blow-ups). **Simplicity:** all else equal, simpler wins — a gain from
deleting code is the best kind; a tiny gain that adds hacky complexity is not worth keeping.

## Selection & decision policy
Judge on the **honest composite** `rmse/1.6 + (1-d1)/0.46 + 0.35·abs_rel` (lower = better). RMSE and
`d1` (= `a1`, the δ<1.25 threshold accuracy — fraction of pixels within ±25% of GT) dominate because
they are honest signals; ABS_REL is directly optimizable by a relative loss → gameable, so it is
discounted. `train.py` selects the best checkpoint by this composite. Never crown a config that wins one
metric while regressing the others; also weigh Pareto behaviour, noise, complexity, and VRAM.

**Noise & confirmation.** σ ≈ 0.008 (20+ reruns), up to ~0.014–0.019 on small samples. Large clear
improvement → provisional PASS; near-noise → confirm required; consistent confirmation → strengthen;
inconsistent → mark uncertain / one more justified confirm. **Never crown a sub-0.015 candidate on fewer
than 3 confirming draws.**

## Hypothesis-driven structure (explore mechanisms, not endless HPO)
Every new/refine/combine study states three levels — do not confuse them:
1. **General hypothesis** — WHY this direction matters (a problem or principle), not a parameter value.
2. **Detailed hypothesis** — HOW the mechanism is expected to work.
3. **Experiment note** — WHAT actually changed in the run (the code/param diff).

Exactly one **type** per experiment: **new** (a qualitatively new mechanism) · **refine** (better
formulation/placement of an existing mechanism) · **tune** (HPO that serves a structural idea) ·
**combine** (merge compatible mechanisms — state why they are complementary first) · **confirm**
(validate a near-noise / one-run conclusion; must reference what it confirms).

**Adaptive HPO ladder** (tuning serves research; don't over-tune a weak idea): screen the structural idea
first; if broken in principle or catastrophically worse → conclude FAIL without mandatory HPO; otherwise
`screen → 3 → 5 → 7 → 10`, each extension justified by accumulated evidence (3 = minimum for a viable
idea; 7 = beats/closely competes with its parent; 10 = exceptional final candidate still improving).

Maintain **lineages** (mechanism families) with their own champions in `studies.json`. A method can FAIL
to become global champion yet PASS as a hypothesis (clear reproducible effect, useful specialist profile,
or recombination value). After each study, pick ONE next action (explore / refine / tune / combine /
confirm), append the conclusion, update state, `git push`, and launch the next.

## State files
- `out/results.tsv` — authoritative per-run log, 7 tab-separated columns:
  `commit  abs_rel  rmse  d1  memory_gb  status  description` (status ∈ `keep`/`discard`/`crash`;
  crashes use `0.0000`; description ≤ ~120 chars, terse — long reasoning goes in the study record).
- `out/hypothesis.tsv` — one short row per study (id, lineage, type, conclusion, best_comp, best_exp,
  best_commit, one-line summary).
- `out/hypothesis_details.tsv` — the long per-study reasoning, keyed by `study_id`.
- `studies.json` — active study, HPO stage, `next_exp_id`, `global_champion`, backlog.
- `utils/research.py` — `status` prints state; `composite --abs_rel A --rmse R --d1 D`; `next-id`.

## Reporting & visualization (`utils/report.py`)
Figures live in `out/display/` (tracked) and are embedded in `README.md`:
- `qualitative` → `qualitative.png`: 7 val scenes × `RGB | GT | batvision | best1 | best2` (best1/best2 =
  your top "my model" checkpoints in `checkpoints/best1|best2/`; missing → "pending" tile; RGB is N/A in
  the simplified dataset).
- `progress` → `score_progress.png`: RMSE, ABS_REL and a1 (d1) each as a full-width graph vs experiment,
  running-best highlighted.
- `readme` → refreshes the `<!-- RESULTS:START/END -->` metrics table in `README.md`.
- `all [--prune]` runs all three. Regenerate after each run.

The **bottom of `README.md` holds a network flowchart** — a Mermaid `flowchart TD` (top-to-bottom
block diagram) with **two separate networks**: `current` (RayDPT, my model) on top and the
`batvision` reference U-Net below (stacked via an invisible `~~~` link). Keep it in sync when the
architecture changes.

## Image retention
Per-epoch dumps in `outputs/<exp>/visualizations/` are **git-ignored** (never committed) and pruned for
disk: `python utils/report.py prune` keeps the earliest (initial) epoch, a few evenly-spaced milestones,
and the latest N (`--keep-latest`, default 6), deleting the rest. Curated figures in `out/display/` are
**tracked** — never prune those.

## Run autonomously & indefinitely
Once the loop begins, do **not** pause to ask "should I keep going?". The moment a run finishes: record
it, regenerate figures, `git push`, and launch the next. Between launches the single GPU is busy ~1 hr/run
— schedule a wakeup to poll and resume the loop across turns; always have the next experiment staged. On
error: log the failure, recover to a valid parent/lineage champion, diagnose briefly, continue. On
plateau: reduce local tuning and move to a different mechanism. The loop runs until the human interrupts.
