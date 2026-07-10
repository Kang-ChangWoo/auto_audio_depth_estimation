#!/usr/bin/env bash
# Stage F2b: fill the 2x2 so the fast knobs can be separated from the lr.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 214>out/.attrib.lock
flock -n 214 || { echo "[attrib] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
echo "[attrib] waiting for the lr sweep to finish... $(date -Is)"
while ! grep -q "LRSWEEP DONE" out/lrsweep_driver.log 2>/dev/null; do sleep 60; done
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# I18 is CONFIRMED: at lr 6e-4 the fast config is perfectly stable (E20's mae never rose between
# epochs; max ratio 0.99) and converges (best epoch 23 of 26). The instability was the optimiser,
# not the architecture and not the coarse-layout auxiliary.
#
# But E20 differs from E11 in TWO ways -- the knobs (win32 3, ffn 2) and the lr (6e-4 vs 1.2e-3) --
# so its +0.0083 against E11 cannot be attributed. The 2x2, judged on the composite:
#
#              lr 1.2e-3        lr 6e-4
#   win5 ffn4  E11  1.9093      E22  ?        <- this run
#   win3 ffn2  E17  DIVERGED    E20  1.9176
#
# E22 vs E20 isolates the KNOBS at matched lr.  E22 vs E11 isolates the LR at matched architecture.
# Without it, "the fast defaults cost 0.0083" is a claim about a confound.
run raydpt_e22_ctrl_lr6e4 --epochs 24 --lr 6e-4 --raydpt-win32 5 --ffn-mult 4

echo "ATTRIB DONE $(date -Is)"
