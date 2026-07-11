#!/usr/bin/env bash
# Stage H2 (I28): Huber-log loss -- centre the near-field bulk WITHOUT loosening the tail.
cd "$(dirname "$0")/.."
exec 223>out/.h2.lock
flock -n 223 || { echo "[h2] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}
# E34 (log_mae) confirmed the median-pull diagnosis -- it centred the 1-2m ratio histogram exactly
# as predicted (0.9-1.0 pile 32.1->29.1%, centre 1.0-1.11 29.2->32.1%) -- but netted -0.0034 on d1
# because it also thickened the OVER-prediction tail (>1.25: 15->16.2%). Centre and tail cancelled.
#
# log_huber is the shape that fixes only what failed: Huber on the log-ratio r = log D - log gt, with
# delta = log(1.25) = 0.223 set EXACTLY at d1's +-25% band. Inside the band it is QUADRATIC, pulling
# harder toward ratio 1 than log_mae (centres the bulk more); outside, it is LINEAR with the same
# slope as log_mae (does NOT loosen the tail further). So it should centre without the tail cost.
#
# Champion arch, control E23 (mae, 1.8962) and E34 (log_mae). PRE-REGISTERED: 1-2m interior d1 must
# improve over BOTH, and the ratio histogram must centre WITHOUT the >1.25 tail growing past E23's
# 15.0%. If the tail still grows, the near-field pull is not cleanly loss-shapeable and the line
# closes. ABS_REL is not evidence.
run raydpt_e35_loghuber --epochs 22 --main-loss log_huber
echo "H2 DONE $(date -Is)"
