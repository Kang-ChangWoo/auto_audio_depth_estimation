#!/usr/bin/env bash
# Stage F2 (I18): the instability is in the trunk, so tune the optimiser -- not the architecture.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 213>out/.lrsweep.lock
flock -n 213 || { echo "[lrsweep] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
echo "[lrsweep] waiting for the stability runs to finish... $(date -Is)"
while ! grep -q "STABLE DONE" out/stable_driver.log 2>/dev/null; do sleep 60; done
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# CORRECTED CAUSALITY. I first blamed the coarse-layout head because `lc` spiked hardest. E18 set
# its weight to zero -- no gradient at all -- and the run still destabilised at epoch 7, with mae
# jumping x1.51. lc is a sigmoid reading m16: an instrument, not a cause. I named a cause from a
# correlation instead of from an intervention.
#
# The trunk is what moves, and the remaining candidate is the optimiser. lr = 3e-4 x (64/16) =
# 1.2e-3 comes from the linear scaling rule, which is a heuristic for convnets under SGD, not a law
# for a residual attention stack. E9/E11/E12 survived it; E15 (larger KV) and E17 (win32=3, ffn=2)
# did not. Gradient clipping is already 1.0.
#
# This is a `tune` serving a structural goal (a fast config that can be experimented on), not a new
# mechanism. Judge each run on: does mae ever jump more than 1.2x between epochs? and the composite.
run raydpt_e20_fast_lr6e4 --epochs 28 --lr 6e-4
run raydpt_e21_fast_lr3e4 --epochs 28 --lr 3e-4

echo "LRSWEEP DONE $(date -Is)"
