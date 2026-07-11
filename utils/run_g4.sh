#!/usr/bin/env bash
# Stage G3 (D13): is the far-field limit the ENCODER's time pooling, or the sensor?
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 220>out/.g4.lock
flock -n 220 || { echo "[g4] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# D13: adding far-field TIME structure to the DECODER has saturated -- E24 helped, but finer volume
# bins (E29, e2), fine-token routing (E30, kv=e3) and dense capacity (E27) all failed to add to it.
# The saturation point is e3's 64 time columns, which the ENCODER pooled 8x down from the STFT's 512.
#
# This run reads the STFT `spec` DIRECTLY (time 512, 2 cm depth spacing), bypassing the encoder's
# time pooling entirely. It is the clean decider:
#   far deciles improve over E24  -> the ENCODER's pooling was the limit; the volume was starved of
#                                    time BEFORE it ever read e3. A real, actionable finding.
#   far deciles saturate again    -> the limit is NOT time resolution at all; it is the SENSOR (I7,
#                                    angular/ranging observability with two mics). The lever then
#                                    moves to the representation, or it is a scoped ceiling.
#
# Fast parent (E24's config), control E24 (1.8987). Pre-registered: the 7-10 m deciles must improve
# over E24, or drop. ABS_REL is not evidence.
run raydpt_e31_ede_raw --epochs 22 --raydpt-win32 3 --ffn-mult 2 --depth-volume True --depth-volume-src raw

echo "G4 DONE $(date -Is)"
