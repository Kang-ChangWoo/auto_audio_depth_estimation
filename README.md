# Auto Audio Depth Estimation

Autonomous research — binaural echoes → ERP radial depth (RayDPT, SoundSpaces).

---

## Status

| | |
|---|---|
| **Phase** | 2026-July · RayDPT-validation |
| **Champion** | *(baseline pending — E0)* |
| **Best composite** | — |
| **Branch (active)** | `2026-July-RayDPT-validation` |
| **Archive (June)** | `2026-June-RayDPT-improvement` — champion composite ~2.030 |

*(This table is the live status board — updated as the phase progresses. `python utils/research.py status` for detail.)*

## Progression (composite, lower = better)

| phase | best | note |
|---|---|---|
| 2026-June (archived) | ~2.030 | multi-res STFT + interaural coherence + TTA |
| 2026-July (this) | — | fresh baseline + fixed coarse/low loss target + de-weighted-abs_rel selection |

## Layout

```
train.py  prepare.py(fixed)  program.md  studies.json
utils/research.py    out/{results.tsv, hypothesis.tsv, hypothesis_details.tsv}
```
