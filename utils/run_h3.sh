#!/usr/bin/env bash
# Stage H3 (I30 + autonomous continuation): per-image scale calibration, then the scope-predicted
# combine, then confirm. Runs unattended -- do not wait between experiments.
cd "$(dirname "$0")/.."
exec 224>out/.h3.lock
flock -n 224 || { echo "[h3] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}
# I30: per-image learned log-scale. Diagnostics refuted every other near-field explanation --
# resolution (oracle 0.98 ceiling), smoothness (RayDPT smoother than batvision), and loss shape
# (H0/H1, three losses). What remains is a pure GLOBAL multiplicative offset: RayDPT sits a few %
# under the +-25% band centre. A per-image scale head is the one mechanism that moves ALL pixels the
# same ratio. Champion arch, control E23 (1.8962).
# PRE-REGISTERED: 1-2m interior ratio histogram must shift bodily toward 1.0-1.11, and interior d1
# must beat E23. If not, the near field is a genuine ceiling.
run raydpt_e36_scalehead --epochs 22 --scale-head True
# confirm draw (program.md: never crown on one)
run raydpt_e37_scalehead_confirm --epochs 22 --scale-head True
echo "H3 DONE $(date -Is)"
