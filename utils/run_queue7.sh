#!/usr/bin/env bash
# Stage 7 (S5/H2): the CONVERGED 2-layer RayDPT -- the only run that can answer D9.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 206>out/.queue7.lock
flock -n 206 || { echo "[queue7] another queue7 runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# D9 says RayDPT's whole deficit is d1 (angle). But no 2-layer RayDPT has ever CONVERGED, so
# d1 has never been read on a finished 2-layer model (E10: 15 epochs, best = last).
#
# MEASURED cost lever (bench, validation overhead included at 16 s/epoch):
#   L2 ffn4 kv=e3 (E10)  231.6 s/ep -> 15.5 ep   5.89M
#   L2 ffn2 kv=e3        229.7 s/ep -> 15.7 ep   5.29M   (FFN is NOT the cost)
#   L2 ffn4 kv=e4        145.8 s/ep -> 24.7 ep   5.89M   <- this: same params as E10
# The cost was never the FFN; it was the 2048-token KV set. Shrinking cr32's KV from e3 (2048
# fine tokens) to e4 (512 coarse ones) is 4x cheaper at IDENTICAL parameter count.
#
# Caveat, stated up front: this is a mechanism change, not a free speedup. The 32-scale rays now
# attend to coarse audio tokens instead of fine ones. cr16 already used e4, and the original code
# calls e4 "a cheap global cue" for the 64-scale, so this makes the pyramid consistent -- but if
# d1 recovers, the credit is shared between "2 layers" and "coarse KV" and must be ablated.
run raydpt_e11_d32L2_kve4 --amp bf16 --decode-scale 32 --ray-cross-layers 2 \
    --cross-kv32 e4 --batch-size 64 --lr 1.2e-3 --epochs 24

echo "QUEUE7 DONE $(date -Is)"
