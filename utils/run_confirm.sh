#!/usr/bin/env bash
# Champion confirm draws: E23 (1.8962) is the number the whole phase conclusion rests on, and it is
# a single draw. program.md: never rest a claim on one draw. Two more draws pin down its variance.
cd "$(dirname "$0")/.."
exec 225>out/.confirm.lock
flock -n 225 || { echo "[confirm] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}
# Exact champion config (defaults ARE E23: decode32, L2, kv=e4, win5, ffn4, batch64, lr3e-4).
# cudnn.benchmark + non-deterministic kernels mean each draw differs; that IS the variance we want.
run raydpt_e38_champ_confirm1 --epochs 22
run raydpt_e39_champ_confirm2 --epochs 22
echo "CONFIRM DONE $(date -Is)"
