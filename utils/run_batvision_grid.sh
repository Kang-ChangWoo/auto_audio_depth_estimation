#!/usr/bin/env bash
# BatVision reference grid: channel count {2, 5} x magnitude compression {log, nolog}.
# 2ch = [L,R]; 5ch = [L,R,ILD,cosIPD,sinIPD]. One 1-hour run each, strictly sequential
# (single GPU -> parallel runs would corrupt the wall-clock TIME_BUDGET comparison).
# NOTE: no `set -u` — conda's binutils activate hook reads unbound vars (ADDR2LINE) and would abort.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss

run() {  # name  use_log  extra_feat_flags...
    local name="$1"; local uselog="$2"; shift 2
    echo "=== $name (use_log=$uselog) $(date -Is) ==="
    python run_base.py --mode train --experiment-name "$name" --use-log "$uselog" "$@" \
        > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

mkdir -p out/logs
ONLY_LR="--feat-ILD False --feat-cosIPD False --feat-sinIPD False"

run batvision_2ch_nolog False $ONLY_LR
run batvision_2ch_log   True  $ONLY_LR
run batvision_5ch_nolog False
run batvision_5ch_log   True

echo "ALL DONE $(date -Is)"
