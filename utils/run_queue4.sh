#!/usr/bin/env bash
# Fourth-stage queue: make RayDPT able to train to 25 epochs (idea I8).
# Waits for run_queue3.sh, then runs the throughput study.
# Separate file on purpose: bash reads a running script incrementally, never edit a live queue.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."

exec 203>out/.queue4.lock
flock -n 203 || { echo "[queue4] another queue4 runner holds the lock; exiting."; exit 0; }
echo "[queue4] single-instance lock acquired (pid $$)"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs

echo "[queue4] waiting for run_queue3.sh to finish... $(date -Is)"
while ! grep -q "QUEUE3 DONE" out/queue3_driver.log 2>/dev/null; do sleep 60; done
echo "[queue4] stage-3 queue finished. starting. $(date -Is)"

run() { local name="$1"; local script="$2"; shift 2
    echo "=== $name START $(date -Is) ==="
    python "$script" --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# E4 fitted only 5 epochs at 713 s/epoch; batvision fits 25 at ~130 s. Under a wall-clock
# budget that gap IS the score (D5). Three changes, none of which touch the mechanism:
#   I9   delete e5..e8 -- 16.78M of 24.44M params, never called, no gradient.
#        VERIFIED output bit-identical. Model 24.44M -> 7.66M.
#   I8a  LocalSphericalAttention gathers neighbours by slicing a once-padded tensor instead
#        of F.unfold, which materialised 1.8 GB for k and v at 64x128 batch 16.
#        VERIFIED mathematically identical (forward and backward).
#   I8b  bf16 autocast + TF32 + cudnn.benchmark. Metrics still computed in fp32.
#
# TWO experiments, not one, because batch size is a separate hypothesis (a bigger batch means
# fewer optimizer steps per epoch, so it needs the linear lr scaling rule and can regress for
# reasons that have nothing to do with throughput).
#
#   E7  pure speed:  batch 16, lr 3e-4 -- exactly E4's optimisation, only faster.
#                    Isolates "is RayDPT merely compute-starved?" If epochs_ran rises and the
#                    best epoch stops being the last, D5 is confirmed.
#   E8  batch tune:  the batch that utils/bench_raydpt.py measured as fastest per epoch, with
#                    lr scaled linearly. A `tune` experiment serving I8, not a new mechanism.
#
# --epochs 25 so the cosine schedule anneals to zero inside the budget rather than being cut
# off mid-schedule (idea I3). If a run still hits the budget before epoch 25, that is itself
# the measurement.

run raydpt_e7_fast train.py --amp bf16 --batch-size 16 --lr 3e-4 --epochs 25

BATCH=16; AMP=bf16
if [ -f out/raydpt_batch.txt ]; then read -r BATCH AMP _ _ < out/raydpt_batch.txt; fi
LR=$(python -c "print(f'{3e-4 * $BATCH / 16:.6g}')")
echo "[queue4] batch tune: batch=$BATCH amp=$AMP lr=$LR (linear scaling from 3e-4 @ batch 16)"
run raydpt_e8_batch${BATCH} train.py --amp "$AMP" --batch-size "$BATCH" --lr "$LR" --epochs 25

echo "QUEUE4 DONE $(date -Is)"
