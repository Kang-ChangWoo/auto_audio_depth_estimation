#!/usr/bin/env python3
"""
research.py - lightweight research-management helper for Auto Audio Depth Estimation.

This is NOT a framework. It is a thin, readable layer over four human-editable files that
add scientific structure on top of the AutoResearch execution loop (train.py + results.tsv):

  results.tsv    - authoritative per-run log (UNCHANGED format; one row per training run)
  hypotheses.tsv - study-level scientific conclusions (general/detailed hypothesis, PASS/FAIL)
  archive.json   - global + per-lineage champions, informative failures, specialists
  studies.json   - active study state + adaptive-HPO progression + next experiment id

Workflow (see program.md for the full protocol):

  general hypothesis -> detailed hypothesis -> experiment note
    -> structural screen -> adaptive HPO (3->5->7->10, justified) -> confirm if near noise
    -> PASS/FAIL conclusion -> archive update -> pick next action (explore/refine/tune/combine/confirm)
    -> continue indefinitely until externally stopped.

Usage:
  python research.py status                       # print current research state
  python research.py composite --abs_rel A --rmse R --d1 D   # honest composite for a run
  python research.py next-id                       # print the next experiment id

Edit the .tsv/.json files directly for anything this CLI does not cover - they are meant to be
read and written by hand. Keep them small and human-readable.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "archive.json")
STUDIES = os.path.join(HERE, "studies.json")
HYPOTHESES = os.path.join(HERE, "hypotheses.tsv")


# --- honest composite (must match archive.json / EXPERIMENTS.md / train.py selection) ---
def composite(abs_rel, rmse, d1):
    """Honest-weighted composite; lower is better. RMSE + d1 dominate (not directly optimised);
    ABS_REL is discounted because the relative loss can game it."""
    return rmse / 1.6 + (1.0 - d1) / 0.46 + 0.3 * (abs_rel / 0.4)


def _load(path):
    with open(path) as f:
        return json.load(f)


def _save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def load_archive():
    return _load(ARCHIVE)


def load_studies():
    return _load(STUDIES)


def status():
    """Print the current research state: champion, active study, HPO ladder position."""
    arc = load_archive()
    st = load_studies()
    gc = arc["global_champion"]
    print("=" * 72)
    print("Auto Audio Depth Estimation - research state")
    print("=" * 72)
    print(f"metric (lower=better): composite = {arc['metric']['composite']}")
    print(f"noise sigma ~= {arc['metric']['noise_sigma_composite']} "
          f"(small-sample up to {arc['metric']['noise_sigma_small_sample']})")
    print("-" * 72)
    if gc:
        print(f"GLOBAL CHAMPION : {gc['exp_id']} ({gc['lineage']}) commit {gc['commit']}")
        print(f"  abs_rel {gc['abs_rel']}  rmse {gc['rmse']}  d1 {gc['d1']}  "
              f"comp {gc.get('composite_mean', gc.get('composite_mean_3draw', gc.get('composite_best_draw')))}  "
              f"confirmed={gc['confirmed']}")
    else:
        print(f"GLOBAL CHAMPION : (none yet) — {arc.get('phase', '')}")
        if arc.get('reset_note'):
            print(f"  reset: {arc['reset_note'][:96]}")
    print("-" * 72)
    print("LINEAGE CHAMPIONS:")
    for lin, c in arc["lineage_champions"].items():
        print(f"  {lin:32s} {c['exp_id']}")
    print("-" * 72)
    a = st["active_study"]
    print(f"ACTIVE STUDY    : {a['study_id']} [{a['type']}] {a['lineage']} "
          f"(status={a['status']})")
    print(f"  HPO stage     : {a['hpo_stage']} (runs used: {a['hpo_runs_used']})")
    print(f"  general       : {a['general_hypothesis'][:88]}...")
    print(f"  next exp id   : {st['next_exp_id']}")
    if a.get("open_risks"):
        print("  open risks    :")
        for r in a["open_risks"]:
            print(f"    - {r[:96]}")
    print("-" * 72)
    print(f"BACKLOG ({len(st.get('backlog', []))}): "
          + "; ".join(f"[{b['type']}] {b['lineage']}" for b in st.get("backlog", [])))
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
