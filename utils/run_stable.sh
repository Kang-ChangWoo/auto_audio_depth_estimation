#!/usr/bin/env bash
# Stage F1: training FRAGILITY is the real obstacle to fast experimentation, not FLOPs.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 212>out/.stable.lock
flock -n 212 || { echo "[stable] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# EVIDENCE: at lr 1.2e-3 (the batch-64 linear-scaled lr), the three runs that left the
# architecture alone were stable (E9, E11, E12); BOTH runs that changed it diverged in the same
# way (E15 kv16=e3, E17 win3+ffn2). In each case the blow-up is entirely `lc`, the 16x32
# coarse-layout term, which then never recovers -- coarse_head is a Conv2d(dim,1,1) + sigmoid
# sitting on m16 = F16 + se4(e4) with NO normalisation, so once it saturates its gradient is ~0.
# E11 survived not because it had margin but because it sat on the edge.
#
# So the obstacle to fast experimentation is not FLOPs; it is that every knob becomes an lr search.
#
# Two candidate fixes, ONE hypothesis each, cheapest first.
#
# E18 DELETE the fragile term. `lc` is the model's only failure point, and I6 already measured the
#     two auxiliaries INERT on the reference: zeroing both moved batvision's composite by +0.0046
#     (below sigma) while removing 58.2% of the gradient. program.md: a gain from deleting code is
#     the best kind. But I6 was measured on batvision, so it MUST be re-checked on RayDPT.
#     Falsification: if the composite degrades by more than sigma, lc is load-bearing for RayDPT
#     even though it was inert for batvision -- a finding, and then E19 is the answer instead.
run raydpt_e18_noaux --epochs 28 --w-coarse-layout 0

# E19 KEEP the term, normalise its input (idea I15). Runs only if E18 shows lc is load-bearing;
#     harmless to run either way as a second data point on stability.
run raydpt_e19_lowaux --epochs 28 --w-coarse-layout 0.1

echo "STABLE DONE $(date -Is)"
