#!/usr/bin/env bash
# Stage 5: RayDPT that can actually converge inside the 1-hour wall-clock budget.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 204>out/.queue5.lock
flock -n 204 || { echo "[queue5] another queue5 runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# MEASURED (utils/bench, real train step, bf16):
#   decode 64, layers 2, batch 16 -> 500.3 s/epoch ->  7.2 epochs/h   (E4's config)
#   decode 32, layers 2, batch 64 -> 216.5 s/epoch -> 16.6 epochs/h
#   decode 32, layers 1, batch 64 -> 147.3 s/epoch -> 24.4 epochs/h   <- this
#
# This is a CAPACITY change, not a speed change, and it is not silent:
#   - decode_scale 32 drops cr64 + lsa64 + refine64 = 54% of forward. Justified by I2's
#     oracle: a PERFECT predictor at 32x64 scores composite 0.3527 while RayDPT sits at
#     2.0471, so the 64-scale buys resolution the model cannot use.
#   - ray_cross_layers 1 halves the audio<->ray cross-attention, the dominant remaining term.
#   - batch 64 with lr scaled linearly (3e-4 * 64/16 = 1.2e-3); epochs 25 so cosine anneals
#     inside the budget (idea I3).
# Ray-conditioning -- per-ray RayBank queries x audio cross-attention, depth decoded per ray
# direction -- is preserved. That is the architectural invariant; decode resolution is not.
#
# E9 is therefore a COMPOUND change and nothing may be attributed to any single part of it.
# It answers exactly one question: does a RayDPT that CONVERGES beat one that does not (D5)?
run raydpt_e9_d32L1_b64 --amp bf16 --decode-scale 32 --ray-cross-layers 1 \
    --batch-size 64 --lr 1.2e-3 --epochs 25

echo "QUEUE5 DONE $(date -Is)"
