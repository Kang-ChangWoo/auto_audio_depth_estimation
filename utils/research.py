#!/usr/bin/env python3
"""
utils/research.py - lightweight research-management helper for Auto Audio Depth Estimation.

NOT a framework. A thin, readable layer over the human-editable state files:

  out/results.tsv            - authoritative per-run log (one row per training run; SHORT descriptions)
  out/hypothesis.tsv         - study-level PASS/FAIL conclusions (one line each)
  out/hypothesis_details.tsv - the full general/detailed hypothesis + scientific conclusion per study
  studies.json               - active study state, champion, adaptive-HPO progression, next experiment id

Usage (run from repo root):
  python utils/research.py status                                  # print current research state
  python utils/research.py composite --abs_rel A --rmse R --d1 D   # honest composite for a run
  python utils/research.py next-id                                 # print the next experiment id

Edit the .tsv/.json files directly for anything this CLI does not cover.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
STUDIES = os.path.join(ROOT, "studies.json")
RESULTS = os.path.join(OUT, "results.tsv")
HYP = os.path.join(OUT, "hypothesis.tsv")
HYP_DETAILS = os.path.join(OUT, "hypothesis_details.tsv")

# Honest-weighted composite (MUST match train.py model selection). Lower is better. RMSE + d1 dominate
# (not directly optimised -> trustworthy); ABS_REL is directly optimisable (gameable) and varies most,
# so it is de-weighted to an effective per-unit coefficient of 0.35 (2026-July).
COMPOSITE_STR = "rmse/1.6 + (1-d1)/0.46 + 0.35*abs_rel"


def composite(abs_rel, rmse, d1):
    return rmse / 1.6 + (1.0 - d1) / 0.46 + 0.35 * abs_rel


def load_studies():
    with open(STUDIES) as f:
        return json.load(f)


def status():
    st = load_studies()
    gc = st.get("global_champion")
    print("=" * 72)
    print(f"Auto Audio Depth Estimation - research state  [{st.get('phase', '')}]")
    print("=" * 72)
    print(f"metric (lower=better): composite = {COMPOSITE_STR}")
    print("-" * 72)
    if gc:
        print(f"GLOBAL CHAMPION : {gc.get('exp_id')} ({gc.get('lineage')}) commit {gc.get('commit')}")
        print(f"  abs_rel {gc.get('abs_rel')}  rmse {gc.get('rmse')}  d1 {gc.get('d1')}  "
              f"comp {gc.get('composite_mean', gc.get('composite'))}")
    else:
        print("GLOBAL CHAMPION : (none yet)")
        if st.get("reset_note"):
            print(f"  {st['reset_note'][:96]}")
    print("-" * 72)
    a = st["active_study"]
    print(f"ACTIVE STUDY    : {a['study_id']} [{a['type']}] {a['lineage']} (status={a['status']})")
    print(f"  hpo stage     : {a.get('hpo_stage')} (runs used: {a.get('hpo_runs_used', 0)})")
    print(f"  general       : {a['general_hypothesis'][:88]}...")
    print(f"  next exp id   : {st['next_exp_id']}")
    print("-" * 72)
    bl = st.get("backlog", [])
    print(f"BACKLOG ({len(bl)}): " + "; ".join(f"[{b['type']}] {b['lineage']}" for b in bl))
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description="Auto Audio Depth Estimation research helper")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("status", help="print current research state")
    sub.add_parser("next-id", help="print the next experiment id")
    c = sub.add_parser("composite", help="honest composite for a run")
    c.add_argument("--abs_rel", type=float, required=True)
    c.add_argument("--rmse", type=float, required=True)
    c.add_argument("--d1", type=float, required=True)
    args = p.parse_args()

    if args.cmd == "composite":
        print(f"{composite(args.abs_rel, args.rmse, args.d1):.4f}")
    elif args.cmd == "next-id":
        print(load_studies()["next_exp_id"])
    else:
        status()


if __name__ == "__main__":
    main()
