#!/usr/bin/env bash
# Stage G1: EchoDelayVolume on the CHAMPION architecture, and a confirm draw.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 217>out/.g1.lock
flock -n 217 || { echo "[g1] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# I19 PASSED its pre-registered falsification, the same test that killed I13. Against its control
# E21, the echo-delay cost volume improved every far decile -- 7-8 m +0.0692, 8-9 m +0.0759,
# 9-10 m +0.0595 -- and the mean predicted depth at GT 8-9 m rose from 4.758 m to 5.238 m. The
# compression it was designed to attack actually loosened. Composite -0.0112, above sigma.
#
# But E24 carried the fast knobs (win32=3, ffn=2), which E23 showed cost +0.0137 at this lr. So the
# mechanism has never been measured on the champion architecture.
#
# E25: champion config (win32=5, ffn=4, lr 3e-4) + depth volume. Its control is E23 (1.8962).
# E26: a second draw of the same config. program.md forbids crowning a candidate on one draw, and
#      this is now the project's only mechanism whose pre-registered prediction was confirmed --
#      exactly the case where a confirm run is worth an hour.
run raydpt_e25_echodelay_champ --epochs 26 --depth-volume True
run raydpt_e26_echodelay_confirm --epochs 26 --depth-volume True

echo "G1 DONE $(date -Is)"
