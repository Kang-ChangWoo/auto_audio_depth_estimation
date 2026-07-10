#!/usr/bin/env bash
# Stage 10 (S7 / I14): route FINE audio tokens to the cheap coarse ray scale.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 209>out/.queue10.lock
flock -n 209 || { echo "[queue10] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# I13 is dropped: BOTH relative dense losses made the far deciles WORSE, which was the
# pre-registered falsification. The DIAGNOSIS survives -- far-field compression is where d1 is
# lost -- but the loss is not its cause.
#
# I14 comes from data already collected. E9 and E12 differ only in which audio tokens the
# 32-scale rays attend to:
#     E9  cr32 on e3 (2048 fine tokens)   d1 @ 8-9 m = 0.2544
#     E12 cr32 on e4 ( 512 coarse tokens) d1 @ 8-9 m = 0.1579
# Fine tokens buy +0.0965 of far-field d1. Physics: a far surface returns a LATE, WEAK echo --
# late frames on the spectrogram's time axis -- and e4 pools that tail away. batvision's U-Net
# skips carry the same detail to its decoder, and it scores 0.4023 there.
#
# But cr32-on-e3 costs 2048x2048 pairs and never converged (E10). Attention cost is
# (queries x kv), so route the fine tokens to the CHEAP scale instead:
#     cr16 on e3 = 512 x 2048 = 1.05M pairs -- the same price as cr32-on-e4, 4x cheaper than E10.
# MEASURED: 169.5 s/epoch = 21.2 epochs. Parameter count unchanged at 5.89M.
#
# PRE-REGISTERED: the 7-10 m deciles must improve over E11. Overall d1 should rise. If the far
# deciles do not improve, I14 is dropped whatever the composite does -- exactly as I13 was.
# RMSE may move either way; ABS_REL is not evidence.
run raydpt_e15_kv16e3 --amp bf16 --decode-scale 32 --ray-cross-layers 2 \
    --cross-kv32 e4 --cross-kv16 e3 --batch-size 64 --lr 1.2e-3 --epochs 21

echo "QUEUE10 DONE $(date -Is)"
