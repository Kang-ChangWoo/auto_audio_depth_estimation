#!/usr/bin/env bash
# Third-stage queue. Waits for run_queue2.sh, then runs the I10 discriminator.
# Separate file on purpose: bash reads a running script incrementally, so a live queue is never edited.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."

exec 202>out/.queue3.lock
flock -n 202 || { echo "[queue3] another queue3 runner holds the lock; exiting."; exit 0; }
echo "[queue3] single-instance lock acquired (pid $$)"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs

echo "[queue3] waiting for run_queue2.sh to finish... $(date -Is)"
while ! grep -q "QUEUE2 DONE" out/queue2_driver.log 2>/dev/null; do sleep 60; done
echo "[queue3] stage-2 queue finished. starting. $(date -Is)"

run() { local name="$1"; local script="$2"; shift 2
    echo "=== $name START $(date -Is) ==="
    python "$script" --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# I10 -- why did E5 (arm A) improve, when its analysis window, and therefore its temporal
# RESOLUTION, was unchanged?
#
# Nyquist says arm A added almost no information: the win=400 window has an envelope
# bandwidth of ~120 Hz, so hop=160 already samples at 1.2x Nyquist and hop=40 is a 4x
# oversample. What hop=40 DID change is the picture: _features resizes the (freq, time)
# grid to (256,512) with NEAREST, so T=18 frames become a staircase of 28.4 px blocks,
# while T=71 gives 7.2 px blocks.
#
# This run holds the information FIXED (win 400, hop 160 -- exactly E2's control) and
# removes only the staircase, by resizing bilinearly instead of nearest.
#   recovers most of E5's RMSE gain  -> the gain was INTERPOLATION SMOOTHNESS, not temporal
#                                       information. E5's mechanism story is wrong, and the
#                                       real defect is the nearest resize in _features.
#   no gain                          -> E5's gain really did come from the extra time samples.
run batvision_5ch_nolog_bilinear run_base.py --use-log False --feat-interp bilinear

echo "QUEUE3 DONE $(date -Is)"
