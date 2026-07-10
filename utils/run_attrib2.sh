#!/usr/bin/env bash
# Stage F2c: complete the 2x2 at lr 3e-4 so the knobs are attributed at the lr we will actually use.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 215>out/.attrib2.lock
flock -n 215 || { echo "[attrib2] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
echo "[attrib2] waiting for run_attrib.sh... $(date -Is)"
while ! grep -q "ATTRIB DONE" out/attrib_driver.log 2>/dev/null; do sleep 60; done
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# The lr ladder is monotone and it kills the linear scaling rule for this model:
#   fast config @ 1.2e-3  DIVERGED          (E17)
#   fast config @ 6e-4    1.9176  stable    (E20)
#   fast config @ 3e-4    1.9099  stable    (E21)  <- the ORIGINAL batch-16 lr
# E21 matches the E11 champion (1.9093) to +0.0007, one tenth of sigma, with 10% fewer parameters
# and 118.8 vs 143.3 s/epoch. So `lr = 3e-4 x (batch/16)` was never needed; the unscaled lr is right.
#
# E22 (running) gives win5/ffn4 @ 6e-4. E23 gives win5/ffn4 @ 3e-4, completing the square at the lr
# we will actually adopt:
#
#              lr 1.2e-3      lr 6e-4      lr 3e-4
#   win5 ffn4  E11 1.9093     E22  ?       E23  ?
#   win3 ffn2  E17 DIVERGED   E20 1.9176   E21 1.9099
#
# E23 vs E21 is the clean answer to "do the fast knobs cost accuracy?" -- same lr, same budget.
run raydpt_e23_ctrl_lr3e4 --epochs 24 --lr 3e-4 --raydpt-win32 5 --ffn-mult 4

echo "ATTRIB2 DONE $(date -Is)"
