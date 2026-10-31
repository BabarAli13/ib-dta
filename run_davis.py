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
# Budgets and the dense control get FRONTIER_SEEDS each. One seed per point
# is not enough: measured on cold_target, seed-to-seed sd is 0.043 MSE while
# the whole sweep from k=2 to k=32 spans 0.116, so a single-seed curve is
# 2.7 sd wide end to end and its shape cannot be distinguished from noise.
FRONTIER_SEEDS = SEEDS[:3]
# random carries none of the claims and is reported only for comparability
# with published work, so it gets fewer seeds than the cold regimes.
RANDOM_SEEDS = SEEDS[:3]


def _tag(prefix, seed):
    """Seed 2048 keeps the bare tag so runs finished before seeds were
    added to the sweep are still recognised and not repeated."""
    return prefix if seed == SEEDS[0] else f"{prefix}_seed{seed}"


def queue():
    """(group, tag, split, [cfg overrides]) in the order worth running."""
    q = []
    # Headline: the two regimes that carry the claims, then random.
    for split in ("cold_target", "cold_drug", "random"):
        for s in (RANDOM_SEEDS if split == "random" else SEEDS):
            q.append(("seeds", f"seed{s}", split, ["SOLVER.SEED", str(s)]))
    # Scaffold uses a byte-identical split to cold_drug on Davis, since all
    # 68 compounds have distinct Bemis-Murcko scaffolds. One run documents
    # that; more would be duplication.
    q.append(("scaffold", f"seed{SEEDS[0]}", "scaffold",
              ["SOLVER.SEED", str(SEEDS[0])]))
    # Frontier: budgets first at the primary seed so the curve appears
    # early, then the replicate seeds.
    for s in FRONTIER_SEEDS:
        for k in (2, 4, 8, 16, 32, 64):
            q.append(("frontier", _tag(f"topk{k}", s), "cold_target",
                      ["IB.SELECTION", "topk", "IB.BUDGET_K", str(k),
                       "IB.ADAPTIVE_BUDGET", "false", "SOLVER.SEED", str(s)]))
        q.append(("frontier", _tag("dense", s), "cold_target",
                  ["IB.SELECTION", "softmax", "SOLVER.SEED", str(s)]))
    # Canonical folds.
    for f in range(5):
        q.append(("benchmark", f"seed{SEEDS[0]}", f"benchmark/fold{f}",
                  ["SOLVER.SEED", str(SEEDS[0])]))
    return q


def done_path(out, split, tag, data="davis"):
    # Must match train_ib.py exactly, which builds its output directory as
    # OUTPUT_DIR/<data>/<split>_<tag> with the split left intact. A split
    # like "benchmark/fold2" therefore lands in a nested directory,
    # results/davis/benchmark/fold2_seed2048. Flattening the slash here
    # looked for results/davis/benchmark_fold2_seed2048, which never
    # exists, so finished benchmark folds were never detected and were
    # re-run every session.
    return os.path.join(out, data, f"{split}_{tag}", "test_metrics.json")


def is_done(out, split, tag, data="davis"):
    p = done_path(out, split, tag, data)
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
                    help="session budget; no run is started that is not "
                         "expected to finish within it")
    ap.add_argument("--est", type=float, default=2.5,
                    help="hours to assume a run takes until one has been "
                         "measured in this session")
    ap.add_argument("--out", default="./result")
    ap.add_argument("--only", nargs="*", default=None,
                    help="one or more of: seeds scaffold frontier benchmark")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    q = [r for r in queue() if not args.only or r[0] in args.only]

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
    # A run that starts just under the deadline still takes its full hour or
    # two, which is how a session overshoots its host's hard cap and gets
    # killed mid-run, wasting that run entirely. So refuse to start a run
    # that is not expected to finish in time. The estimate starts at
    # args.est and is replaced by this session's own measured mean as soon
    # as one run completes.
    est = args.est * 3600
    durations = []
    for group, tag, split, over in q:
        if is_done(args.out, split, tag):
            skipped += 1
            continue
        left = deadline - time.time()
        if left <= 0:
            print(f"\n  session budget reached. {ran} run(s) this session.")
            print("  start a new session and run the same command to resume.")
            break
        if left < est:
            print(f"\n  {left / 60:.0f} min left, a run needs about "
                  f"{est / 60:.0f} min. Stopping here rather than starting "
                  f"one that would be killed mid-run.")
            print(f"  {ran} run(s) this session. Resume with the same command.")
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
        # Unbuffered child stdout, so progress reaches the log as it
        # happens rather than in 8 KB blocks.
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            # One bad config should not cost the rest of the session.
            print(f"  FAILED (exit {r.returncode}); continuing", flush=True)
            continue
        ran += 1
        dt = time.time() - t0
        durations.append(dt)
        # Use the slowest run seen, not the mean: underestimating is what
        # gets a run killed, and the budget sweep's small-k runs finish much
        # faster than a full-support one.
        est = max(durations)
        print(f"  finished in {dt / 60:.1f} min "
              f"(estimate for the next run: {est / 60:.0f} min)", flush=True)
    else:
        print(f"\n  queue complete. {ran} run(s) this session, "
              f"{skipped} already done.")


if __name__ == "__main__":
    main()
