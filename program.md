# autoresearch

This is an experiment to have the depth estimation network from echoes and FOA(first-order Ambisonics) do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch_audio/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that the dataset directory exists. If not, tell the human to run `conda activate ss && python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 1 hour** (`TIME_BUDGET = 3600` in `train.py`, wall clock training time, excluding startup/compilation). Training automatically stops when the budget is reached. You launch it simply as: `conda activate ss && python train.py --mode train`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation and data loading.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `compute_errors` function in `prepare.py` is the ground truth metric.

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

**The goal is simple: get the lowest ABS_REL and RMSE errors.** Since the time budget is fixed, you don't need to worry about training time — it's always 30 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

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

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	abs_rel	rmse	d1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. ABS_REL achieved (e.g. 0.5889) — use 0.0000 for crashes
3. RMSE achieved (e.g. 1.1503) — use 0.0000 for crashes
4. d1 (delta < 1.25) accuracy (e.g. 0.4782) — use 0.0000 for crashes
5. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

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
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
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

**Keep `results.tsv` and `EXPERIMENTS.md` current** and `git push` to the remote (master) after each experiment, with multiple commits.
