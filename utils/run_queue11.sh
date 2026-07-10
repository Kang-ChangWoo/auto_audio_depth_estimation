#!/usr/bin/env bash
# Stage 11 (S7 retry): E15 diverged at the parent's lr. Re-run at half lr WITH a matched control.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 210>out/.queue11.lock
flock -n 210 || { echo "[queue11] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# The first E15 DIVERGED at epoch 4 (train loss 0.163 -> 0.363 -> 0.576; val d1 0.509 -> 0.331).
# The blow-up was entirely in `lc`, the 16x32 coarse-layout term (0.4402 vs E11's 0.0618), which
# is bounded by 1.0 -- so D_coarse had SATURATED. Routing e3's 2048 tokens into cr16 makes F16 a
# sum over 4x more values; CrossBlock is a residual (q + attn), and coarse_head + sigmoid sits
# directly on m16 = F16 + se4(e4). At lr 1.2e-3 (E11's linear-scaled lr) it saturates.
#
# Classification (program.md): OPTIMISATION failure, not "mechanism unsupported". Nothing is
# concluded about I14 from it.
#
# Halving the lr fixes the instability but would confound E15 against its parent, which trained
# at 1.2e-3. So run a MATCHED CONTROL at the same lr. The comparison is E16 vs E15, not E11 vs E15.
#
#   E16  control: E11's architecture      @ lr 6e-4
#   E15  treatment: + --cross-kv16 e3     @ lr 6e-4
#
# PRE-REGISTERED, unchanged from the run that was killed: the 7-10 m deciles must improve over the
# CONTROL, or I14 is dropped whatever the composite does -- exactly as I13 was. ABS_REL is not
# evidence. Report epochs_ran and best_epoch; the two arms differ in s/epoch (145 vs 190), which is
# itself a confound (D2/D5).
run raydpt_e16_ctrl_lr6e4 --amp bf16 --decode-scale 32 --ray-cross-layers 2 \
    --cross-kv32 e4 --batch-size 64 --lr 6e-4 --epochs 24

run raydpt_e15b_kv16e3_lr6e4 --amp bf16 --decode-scale 32 --ray-cross-layers 2 \
    --cross-kv32 e4 --cross-kv16 e3 --batch-size 64 --lr 6e-4 --epochs 19

echo "QUEUE11 DONE $(date -Is)"
