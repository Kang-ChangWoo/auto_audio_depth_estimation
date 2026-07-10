#!/usr/bin/env bash
# New phase (2026-July-RayDPT-fast): validate the fast default before adopting it.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 211>out/.fastbase.lock
flock -n 211 || { echo "[fastbase] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# The new defaults are the E11 champion plus two knobs the GPU profiler picked out:
#   raydpt_win32 5 -> 3   local spherical attention at 32x64: 25 offsets -> 9.  1.20x
#   ffn_mult     4 -> 2   CrossBlock FFN expansion.                            1.10x on top
# Together 1.32x: 169.1 -> 127.7 s/epoch, 21.3 -> 28.2 epochs/h, 5.89M -> 5.29M params.
#
# BOTH are CAPACITY cuts, so speed alone proves nothing. E17 is the scored run that asks whether
# the fast default costs accuracy. Its parent is E11 (composite 1.9093, d1 0.5710, 23 epochs).
#
# DECISION RULE: adopt the fast default iff its composite is within sigma (0.008) of E11's, OR
# better. The extra epochs it fits are a legitimate part of a wall-clock benchmark, so a WIN
# here does not prove the knobs are free -- it proves the trade is favourable at this budget.
# If it loses by more than sigma, revert win32/ffn to E11's values and keep only the defaults.
run raydpt_e17_fastbase --epochs 28

echo "FASTBASE DONE $(date -Is)"
