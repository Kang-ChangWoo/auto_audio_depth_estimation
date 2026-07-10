#!/usr/bin/env bash
# Stage G0 (I19): echo-delay cost volume -- a STRUCTURAL change, not a loss change.
# NOTE: no `set -u` -- conda's binutils activate hook reads unbound vars (ADDR2LINE).
cd "$(dirname "$0")/.."
exec 216>out/.ede.lock
flock -n 216 || { echo "[ede] another runner holds the lock; exiting."; exit 0; }
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate ss
mkdir -p out/logs
echo "[ede] waiting for the attribution runs... $(date -Is)"
while ! grep -q "ATTRIB2 DONE" out/attrib2_driver.log 2>/dev/null; do sleep 60; done

# measure cost before spending an hour on it
echo "[ede] benchmarking..."
python - <<'PY'
import torch,time,sys; sys.path.insert(0,'.')
from utils.evallock import eval_lock
import train
dev=torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.backends.cudnn.benchmark=True
with eval_lock('bench_ede'):
    for dvol in (False, True):
        sys.argv=['train.py','--mode','train','--depth-volume',str(dvol)]
        cfg=train.make_config(train.parse_args()); mcfg=train.build_model_cfg(cfg)
        m=train.build_model(cfg).to(dev).train(); opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
        B=cfg.mode.batch_size
        x=torch.randn(B,5,256,512,device=dev); gt=torch.rand(B,1,256,512,device=dev); mask=(gt>0.05).float()
        torch.cuda.reset_peak_memory_stats()
        for i in range(8):
            if i==2: torch.cuda.synchronize(); t0=time.time()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=m(x); loss,_=train.composite_loss(out,gt,mask,mcfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        torch.cuda.synchronize(); dt=(time.time()-t0)/6; pk=torch.cuda.max_memory_allocated()/1e9
        se=dt*(28800/B)+16
        print(f"  depth_volume={str(dvol):5s}  {dt*1e3:6.1f} ms/step  {pk:5.1f} GB  {se:6.1f} s/ep  {3600/se:4.1f} ep/h")
        del m,opt,x,gt,mask,out,loss; torch.cuda.empty_cache()
PY

run() { local name="$1"; shift
    echo "=== $name START $(date -Is) ==="
    python train.py --mode train --experiment-name "$name" "$@" > "out/logs/${name}.log" 2>&1
    echo "=== $name exit=$? $(date -Is) ==="
}

# I19 -- echo-delay cost volume. The encoder's WIDTH axis is time: spec is (freq 256, time 512),
# so e3 is (freq 32, time 64) and its 64 columns are depth hypotheses spanning 0.08-9.92 m via
# d = c*t/2. Today the model must LEARN that correspondence. This gives it.
#
# For each ray r and hypothesis j: the ray query attends over FREQUENCY within column j alone
# (azimuth lives in the per-frequency ILD/IPD; time must not be mixed across hypotheses), a small
# MLP scores each hypothesis, and a softmax over the DEPTH axis yields p(d | ray). Then
# depth_r = sum_j p_j d_j: a soft-argmax over echo delay. +34k parameters.
#
# It attacks two MEASURED failures at once:
#   far-field range compression -- at GT 8-9 m RayDPT predicts 4.97 m and scores d1 0.1751 against
#     batvision's 0.4023. A distribution over depth bins cannot collapse to a conditional median,
#     and the evidence for a far surface sits in a LATE column the structure now points at.
#   the fragile coarse head -- a sigmoid one 1x1 conv from the deepest attention output, which
#     saturated and diverged (D11). Replaced by a softmax over physically meaningful bins.
#
# CONTROL is E21 (1.9099): the current defaults are exactly its config, so this run differs from it
# in ONE thing.
#
# PRE-REGISTERED FALSIFICATION, the same test that killed I13: the 7-10 m deciles must improve over
# E21. If they do not, I19 is dropped whatever the composite does. Overall d1 must clear sigma
# (0.008) to be crowned. ABS_REL is not evidence.
run raydpt_e24_echodelay --epochs 28 --depth-volume True

echo "EDE DONE $(date -Is)"
