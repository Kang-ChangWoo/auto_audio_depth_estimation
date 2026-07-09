# Auto Audio Depth Estimation — Experiment Findings (2026-July RayDPT-validation)

*(Project: **Auto Audio Depth Estimation** / `auto-audio-depth-estimation`. Model: **RayDPT**. This is a
**FRESH RESET** — the running per-experiment log for the 2026-July **validation** phase. Study-level
conclusions live in `hypotheses.tsv` / `archive.json`; see `program.md` → "Research workflow (v2)".)*

Audio → ERP radial depth (SoundSpaces, 256×512). Fixed **1-hour** training budget per run.
Metric: `compute_errors` in `prepare.py` — **ABS_REL, RMSE, d1 (δ<1.25)**. Live log: `results.tsv`.

## Why this reset

The **June improvement phase** reached a strong champion — **12ch multi-res STFT + 2-scale interaural
coherence + TTA, composite ~2.030** (ABS_REL −26% / RMSE ~−9% / d1 +7pts vs baseline) — fully archived on
branch **`2026-June-RayDPT-improvement`** (164 runs, 29 studies, commit `ff22f23`).

This **`2026-July-RayDPT-validation`** phase restarts from the **original RayDPT baseline** (`f677b0f`,
5ch, ~0.4434 ABS_REL) with a **corrected loss**, to re-validate the findings.

### The loss fix (the "invalid metric")

The auxiliary **coarse-layout** and **low-pass** loss *targets* were computed by naive `adaptive_avg_pool2d(gt)`
/ `gaussian_blur_erp(gt)` — which **averaged/blurred `gt` over ALL pixels, including invalid ones (`gt=0`
where `mask=0`)**. This diluted the target toward 0 in any cell/region containing invalid pixels, so the
coarse & low-pass losses were trained against a **corrupted target**. Fixed to **mask-weighted pooling**:
`target = sum(gt·mask)/sum(mask)` (mean of *valid* depths only). `compute_errors` (the eval metric) is
**unchanged**; only the training loss targets are corrected.

## Baseline & plan

Fresh baseline = original RayDPT (5ch [logL,logR,ILD,cosIPD,sinIPD] + flip-aug, no TTA, no FEATURE_FN cues,
24.44M params) + the loss fix. The framework is unchanged (`program.md` v2, `research.py`, the
hypothesis-driven workflow; `prepare.py` FEATURE_FN/EMIT_RAW seams present but gated off → byte-identical).

Plan: run E0 (baseline), then follow the workflow to **re-validate the June findings under the corrected
loss** — optimisation envelope (bf16/bs32/lr/EMA/cosine), coarse/geometry ray attention, gated skips,
simplification (drop the pix2pix tail), multi-res STFT, interaural coherence, TTA. The corrected coarse/low
targets especially affect the coarse-layout loss that the geometry/coarse-reasoning studies leaned on, so
those conclusions may shift.

## Results so far (2026-July)

*(none yet — E0 baseline pending)*

| run | change | ABS_REL | RMSE | d1 | verdict |
|---|---|---|---|---|---|
