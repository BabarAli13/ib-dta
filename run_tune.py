"""
Hyperparameter study, selected on validation only.

The forty-run Davis campaign used the DrugBAN defaults unchanged, and the
benchmark folds that produced our headline accuracy had best epochs of 37,
37, 39, 39 and 46 against a ceiling of 50. With patience 12, a run peaking
at 39 or later was stopped by the ceiling while still improving. Those
numbers are under-trained, not converged, so the first thing this study
does is remove the ceiling.

    python run_tune.py --list
    python run_tune.py --hours 24 --out ./tune
    python run_tune.py --report --out ./tune

Method
------
One factor at a time from a baseline, rather than a grid: with runs this
long a grid is unaffordable and a one-factor sweep identifies the binding
constraint, which is what we need.

Selection is on **validation MSE only**, read from each run's history.csv.
This script never reads test_metrics.json. Tuning against a test set and
then reporting that same test set is the most common way a benchmark
number becomes meaningless, and the separation is enforced here rather
than left to discipline.

The study runs on Davis benchmark/fold0, which has its own validation
split and shares the held-out test set with the other folds. The winning
configuration is then applied to all five folds, and only then is test
touched.
"""
import argparse
import json
import os
import subprocess
import sys
import time

CFG = "configs/IBDTA.yaml"
DATA = "davis"
SPLIT = "benchmark/fold0"

# Everything is measured against this. It is the shipped configuration
# with the epoch ceiling lifted, because the ceiling is known to bind and
# leaving it in place would confound every other comparison.
BASE = ["SOLVER.MAX_EPOCH", "250", "SOLVER.PATIENCE", "25"]

# (tag, overrides on top of BASE). Ordered by expected return: the epoch
# ceiling first, then the loss weighting, then capacity, then the
# optimiser.
TRIALS = [
    ("base",        []),
    # The ranking term directly optimises the ordering that CI measures,
    # and CI is where we trail the published models. Our own earlier work
    # found removing this term cost more CI than removing four curated
    # biological-hint pathways.
    ("rank0.3",     ["LOSS.LAMBDA_RANK", "0.3"]),
    ("rank0.8",     ["LOSS.LAMBDA_RANK", "0.8"]),
    # 0.42 M parameters against roughly 1.5 M for DrugBAN.
    ("hdim192",     ["IB.H_DIM", "192"]),
    ("hdim256",     ["IB.H_DIM", "256"]),
    # If the ceiling was wrong then the cosine schedule was decaying over
    # the wrong horizon too.
    ("lr1e-3",      ["SOLVER.LR", "0.001"]),
    ("lr2e-4",      ["SOLVER.LR", "0.0002"]),
    # Davis kinases average about 790 residues; padding to 1200 spends
    # compute on padding and dilutes the protein representation.
    ("plen800",     ["PROTEIN.MAX_LEN", "800"]),
    # Sparsity pressure shapes the support size and may be over-tight.
    ("sparse1e-4",  ["LOSS.LAMBDA_SPARSE", "0.0001"]),
]


def out_dir(out, tag):
    return os.path.join(out, DATA, f"{SPLIT}_{tag}")


def val_of(out, tag):
    """Best validation MSE for a finished trial, or None.

    Deliberately reads history.csv and not test_metrics.json: the test set
    must not influence which configuration we choose.
    """
    h = os.path.join(out_dir(out, tag), "history.csv")
    if not os.path.exists(h):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(h)
        if df.empty or "val_mse" not in df:
            return None
        return {"val_mse": float(df.val_mse.min()),
                "val_ci": float(df.loc[df.val_mse.idxmin(), "val_ci"]),
                "best_epoch": int(df.loc[df.val_mse.idxmin(), "epoch"]),
                "epochs_run": int(df.epoch.max())}
    except Exception:
        return None


def is_done(out, tag):
    # A trial counts as done only once training has finished, which is
    # when last.pt has been removed.
    d = out_dir(out, tag)
    return (os.path.exists(os.path.join(d, "history.csv"))
            and not os.path.exists(os.path.join(d, "last.pt")))


def report(out):
    rows = []
    for tag, over in TRIALS:
        v = val_of(out, tag)
        rows.append((tag, over, v))
    done = [r for r in rows if r[2]]
    print(f"\n  {'trial':<12} {'val MSE':>9} {'val CI':>8} "
          f"{'best ep':>8} {'ran':>5}   overrides")
    for tag, over, v in rows:
        if v is None:
            print(f"  {tag:<12} {'--':>9} {'--':>8} {'--':>8} {'--':>5}   "
                  f"{' '.join(over) or '(baseline)'}")
        else:
            print(f"  {tag:<12} {v['val_mse']:>9.4f} {v['val_ci']:>8.4f} "
                  f"{v['best_epoch']:>8d} {v['epochs_run']:>5d}   "
                  f"{' '.join(over) or '(baseline)'}")
    if not done:
        print("\n  nothing finished yet")
        return
    base = next((r for r in done if r[0] == "base"), None)
    best = min(done, key=lambda r: r[2]["val_mse"])
    print(f"\n  best on validation: {best[0]} at {best[2]['val_mse']:.4f}")
    if base and best[0] != "base":
        d = base[2]["val_mse"] - best[2]["val_mse"]
        print(f"  improvement over baseline: {d:+.4f} MSE "
              f"({d / base[2]['val_mse'] * 100:+.1f}%)")
    if best[2]["best_epoch"] > best[2]["epochs_run"] - 10:
        print("  WARNING: the winner peaked near the end of its run and may "
              "still be improving. Raise MAX_EPOCH again before trusting it.")
    print("\n  Test set untouched. Apply the winner to all five folds with "
          "run_davis.py before reading any test number.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--est", type=float, default=3.0,
                    help="hours to assume a trial takes until measured")
    ap.add_argument("--out", default="./tune")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    trials = [t for t in TRIALS if not args.only or t[0] in args.only]

    if args.report:
        report(args.out)
        return
    if args.list:
        for tag, over in trials:
            mark = "done" if is_done(args.out, tag) else "    "
            print(f"  [{mark}]  {tag:<12} {' '.join(over) or '(baseline)'}")
        pend = sum(1 for t, _ in trials if not is_done(args.out, t))
        print(f"\n  {len(trials) - pend}/{len(trials)} done, {pend} remaining")
        return

    if not os.path.isdir(os.path.join("datasets", DATA, "benchmark")):
        print("  building canonical folds first")
        subprocess.run([sys.executable, "benchmark_split.py",
                        "--datasets", DATA], check=True)

    deadline = time.time() + args.hours * 3600
    est = args.est * 3600
    durations, ran = [], 0
    for tag, over in trials:
        if is_done(args.out, tag):
            continue
        left = deadline - time.time()
        if left < est:
            print(f"\n  {left / 60:.0f} min left, a trial needs about "
                  f"{est / 60:.0f} min. Stopping here.")
            break
        cmd = [sys.executable, "train_ib.py", "--cfg", CFG, "--data", DATA,
               "--split", SPLIT, "--tag", tag, "--no-audit",
               "--set", "RESULT.OUTPUT_DIR", args.out, *BASE, *over]
        print(f"\n{'=' * 68}\n  trial {tag}   {' '.join(over) or '(baseline)'}"
              f"   ({left / 3600:.1f} h left)\n{'=' * 68}", flush=True)
        t0 = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}); continuing", flush=True)
            continue
        ran += 1
        durations.append(time.time() - t0)
        est = max(durations)
        v = val_of(args.out, tag)
        if v:
            print(f"  {tag}: val MSE {v['val_mse']:.4f} at epoch "
                  f"{v['best_epoch']} of {v['epochs_run']}", flush=True)
    report(args.out)


if __name__ == "__main__":
    main()
