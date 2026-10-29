"""
Queue every Davis run the paper needs, resumably.

Designed for session-limited free GPU (Kaggle 12 h, Colab less). Run the
same command in each new session: finished runs are detected by their
test_metrics.json and skipped, so the queue advances instead of restarting.

    python run_davis.py --hours 11          # fill one Kaggle session
    python run_davis.py --list              # show the queue and what is done
    python run_davis.py --only frontier     # one group at a time

Results land in RESULT.OUTPUT_DIR (./result by default). On Kaggle or
Colab, point that at persistent storage:

    python run_davis.py --hours 11 \
        --out /kaggle/working/results

Groups
------
seeds      IB-DTA on each split regime, 5 seeds. Fills the headline table.
scaffold   One run. Davis scaffolds are all distinct, so this should
           reproduce cold_drug exactly; the run is the evidence for that
           claim, not an independent regime.
frontier   Budget sweep k = 2..64 plus the dense control. Fills the
           faithfulness-accuracy frontier figure.
benchmark  Canonical DeepDTA folds, for numbers comparable with published
           tables rather than with our own retraining.
"""
import argparse
import json
import os
import subprocess
import sys
import time

CFG = "configs/IBDTA.yaml"
SEEDS = [2048, 7, 13, 101, 999]


def queue():
    """(group, tag, split, [cfg overrides]) in the order worth running."""
    q = []
    # Headline: the two regimes that carry the claims, then random for
    # comparability with published work.
    for split in ("cold_target", "cold_drug", "random"):
        for s in SEEDS:
            q.append(("seeds", f"seed{s}", split, ["SOLVER.SEED", str(s)]))
    # Scaffold is redundant on Davis; one run documents that.
    q.append(("scaffold", f"seed{SEEDS[0]}", "scaffold",
              ["SOLVER.SEED", str(SEEDS[0])]))
    # Frontier: hard budgets, then the dense control.
    for k in (2, 4, 8, 16, 32, 64):
        q.append(("frontier", f"topk{k}", "cold_target",
                  ["IB.SELECTION", "topk", "IB.BUDGET_K", str(k),
                   "IB.ADAPTIVE_BUDGET", "false", "SOLVER.SEED", str(SEEDS[0])]))
    q.append(("frontier", "dense", "cold_target",
              ["IB.SELECTION", "softmax", "SOLVER.SEED", str(SEEDS[0])]))
    # Canonical folds.
    for f in range(5):
        q.append(("benchmark", f"seed{SEEDS[0]}", f"benchmark/fold{f}",
                  ["SOLVER.SEED", str(SEEDS[0])]))
    return q


def done_path(out, split, tag):
    return os.path.join(out, "davis", f"{split.replace('/', '_')}_{tag}",
                        "test_metrics.json")


def is_done(out, split, tag):
    p = done_path(out, split, tag)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            json.load(f)
        return True
    except (ValueError, OSError):
        return False       # truncated by a killed session; redo it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=11.0,
                    help="stop launching new runs after this long")
    ap.add_argument("--out", default="./result")
    ap.add_argument("--only", default=None,
                    help="seeds | scaffold | frontier | benchmark")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    q = [r for r in queue() if args.only in (None, r[0])]

    if args.list:
        pend = 0
        for group, tag, split, _ in q:
            ok = is_done(args.out, split, tag)
            pend += 0 if ok else 1
            print(f"  [{'done' if ok else '    '}]  {group:9s} "
                  f"{split:18s} {tag}")
        print(f"\n  {len(q) - pend}/{len(q)} done, {pend} remaining "
              f"(~{pend * 2.5:.0f} h on a T4)")
        return

    if not os.path.isdir(os.path.join("datasets", "davis")):
        raise SystemExit(
            "datasets/davis missing. Run first:\n"
            "  python download_data.py --datasets davis\n"
            "  python splits.py --csv datasets/davis/full.csv "
            "--out datasets/davis")
    if any(r[2].startswith("benchmark/") for r in q) and not os.path.isdir(
            os.path.join("datasets", "davis", "benchmark")):
        print("  building canonical folds first")
        subprocess.run([sys.executable, "benchmark_split.py",
                        "--datasets", "davis"], check=True)

    deadline = time.time() + args.hours * 3600
    ran = skipped = 0
    for group, tag, split, over in q:
        if is_done(args.out, split, tag):
            skipped += 1
            continue
        if time.time() > deadline:
            print(f"\n  session budget reached. {ran} run(s) this session.")
            print("  start a new session and run the same command to resume.")
            break
        cmd = [sys.executable, "train_ib.py", "--cfg", CFG,
               "--data", "davis", "--split", split, "--tag", tag,
               "--set", "RESULT.OUTPUT_DIR", args.out, *over]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        print(f"\n{'=' * 68}\n  {group}  {split}  {tag}"
              f"   ({(deadline - time.time()) / 3600:.1f} h left)\n{'=' * 68}",
              flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            # One bad config should not cost the rest of the session.
            print(f"  FAILED (exit {r.returncode}); continuing", flush=True)
            continue
        ran += 1
        print(f"  finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        print(f"\n  queue complete. {ran} run(s) this session, "
              f"{skipped} already done.")


if __name__ == "__main__":
    main()
