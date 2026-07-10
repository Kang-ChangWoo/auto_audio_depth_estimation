#!/usr/bin/env bash
# Stage G2 (DC2): extend the ONE confirmed mechanism. EchoDelayVolume at finer time resolution,
# and its scope-predicted combine. Both on the FAST parent, where I19 is known to help.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 219>out/.g3.lock
flock -n 219 || { echo "[g3] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# CONTEXT. Champion E23 (1.8962) trails batvision (1.8567) by 0.0395, essentially all d1, and the
# d1 deficit is concentrated in the far field (7-10 m: +0.12 to +0.17 per decile). Three independent
# results say fine TIME resolution is the lever, and EchoDelayVolume -- the only mechanism whose
# pre-registered prediction was confirmed -- reads e3, the STFT's 512 time columns pooled 8x.
#
# The FAST config is the parent here (win32=3, ffn=2, lr 3e-4): E24 showed EchoDelayVolume helps it,
# and its extra capacity does not already saturate the far field. Control for both = E24 (1.8987).
# --epochs 24 so cosine anneals inside the budget on the fast (faster) architecture.
#
# H8: read e2 (time 128) instead of e3 (time 64) -- double the echo-delay resolution, ~free.
run raydpt_e29_ede_e2 --epochs 24 --raydpt-win32 3 --ffn-mult 2 --depth-volume True --depth-volume-src e2

# H9: the combine I19's scope predicts. EchoDelayVolume + fine-token routing (cross_kv32=e3): both
# feed far-field signal by DIFFERENT paths -- the volume reads time columns, kv=e3 gives cr32 the
# 2048 fine tokens. kv=e3 diverged at lr 1.2e-3 (D11); at 3e-4 (I18) it should be stable.
run raydpt_e30_ede_kve3 --epochs 22 --raydpt-win32 3 --ffn-mult 2 --depth-volume True --cross-kv32 e3

echo "G3 DONE $(date -Is)"
