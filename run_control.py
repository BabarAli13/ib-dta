"""
The controlled comparison: same everything, different decoder.

Table 3 of the paper asks what the additive decode costs against the
bilinear contraction it replaces, and until now the paper could only
bound it. The bound is loose because the only pooled number available was
DrugBAN's published one, which comes from a different split protocol, a
different training budget and a different codebase, so the 0.124 MSE
between us and the published field is confounded with all three.

This runs the pooled decoder on the same encoders, the same loader, the
same splits, the same optimiser, the same schedule and the same stopping
rule, so that `IB.DECODER` is the only difference between the two rows.
That turns the bound into a measurement.

Two configurations per fold, matching the two IB-DTA rows already in
Table 1: the shipped 128-wide one and the 256-wide one the sweep
selected. Running only the 256 pair would leave open whether the pooled
decoder happens to prefer a different capacity, which is exactly the kind
of hole a reviewer should find.

    python run_control.py              # the queue
    python run_control.py --list       # what is measured so far

The deletion audit runs on the pooled model too, and that is the second
reason this exists. The pooled model has no additive decomposition, so
the audit is applied to its attention map, which is the object a reader
of a published heatmap is invited to interpret. What comes back is the
measured infidelity of that heatmap: AFS says whether deleting the
contacts it highlights matters more than deleting random ones, and the
residual says how far the prediction moves from what the map implies. For
the additive decoder both are settled by algebra. For this one they are
empirical, and the comparison is the paper's central claim stated as two
numbers rather than as an argument.
"""
import argparse
import json
import os
import statistics
import sys
import time

from keepawake import keep_awake
from run_tune import BASE, _enable_ansi, _hms, run_trial

CFG = "configs/IBDTA.yaml"
DATA = "davis"
FOLDS = [0, 1, 2, 3, 4]
SEED = "2048"

MAX_EPOCH = int(BASE[BASE.index("SOLVER.MAX_EPOCH") + 1])
PATIENCE = int(BASE[BASE.index("SOLVER.PATIENCE") + 1])

# (tag, overrides). The additive rows are already measured by
# run_final.py and run_davis.py; these are the controls for them.
ARMS = [
    ("pooled128", ["IB.DECODER", "pooled", "IB.H_DIM", "128"]),
    ("pooled256", ["IB.DECODER", "pooled", "IB.H_DIM", "256"]),
]


def split_of(f):
    return f"benchmark/fold{f}"


def out_dir(out, f, tag):
    return os.path.join(out, DATA, f"{split_of(f)}_{tag}")


def is_done(out, f, tag):
    p = os.path.join(out_dir(out, f, tag), "test_metrics.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as fh:
            json.load(fh)
        return True
    except (ValueError, OSError):
        return False       # truncated by a hard power loss; redo it


def summarise(out, tag):
    """Mean and spread over whichever folds exist, or None if none do."""
    rows = []
    for f in FOLDS:
        p = os.path.join(out_dir(out, f, tag), "test_metrics.json")
        if os.path.exists(p):
            try:
                rows.append(json.load(open(p)))
            except (ValueError, OSError):
                pass
    if not rows:
        return None

    def agg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        return (statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0)

    return {"n": len(rows), "mse": agg("mse"), "ci": agg("ci"),
            "support": agg("support"), "afs": agg("afs"),
            "resid": max((r.get("relative_residual") or 0.0) for r in rows)}


def report(out):
    print(f"\n  {'arm':<12}{'n':>3}{'test MSE':>18}{'CI':>16}"
          f"{'support':>11}{'AFS':>9}{'max resid':>12}")
    print(f"  {'-' * 79}")
    for tag, _ in ARMS:
        s = summarise(out, tag)
        if not s:
            print(f"  {tag:<12}{'-':>3}   pending")
            continue
        mu, sd = s["mse"]
        cmu, csd = s["ci"]
        smu, _ = s["support"]
        amu, _ = s["afs"]
        print(f"  {tag:<12}{s['n']:>3}"
              f"{mu:>11.4f} +/-{sd:>5.4f}"
              f"{cmu:>10.3f} +/-{csd:>5.3f}"
              f"{smu:>11.1f}{amu:>+9.3f}{s['resid']:>12.1e}")
    print("\n  Compare against the additive rows from run_final.py --list.")
    print("  The MSE difference is the price of exact attribution; the AFS")
    print("  and residual columns are what that price buys.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./result")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="session budget; 0 means no limit")
    ap.add_argument("--est", type=float, default=4.0,
                    help="hours to assume a run takes until one is measured")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these arms, e.g. --only pooled256")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--log", default="control_log.txt")
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()

    if args.list:
        report(args.out)
        return

    _enable_ansi()
    keep_awake()
    live = not args.no_live and sys.stdout.isatty()

    arms = [a for a in ARMS if not args.only or a[0] in args.only]
    jobs = [(tag, over, f) for tag, over in arms for f in FOLDS
            if not is_done(args.out, f, tag)]
    total = len(arms) * len(FOLDS)
    print(f"  controlled comparison, pooled decoder, seed {SEED}")
    print(f"  {total - len(jobs)}/{total} runs done, {len(jobs)} to go "
          f"(~{len(jobs) * args.est:.0f} h)")
    if not jobs:
        report(args.out)
        return

    deadline = time.time() + args.hours * 3600 if args.hours else None
    est, durations, t0 = args.est * 3600, [], time.time()
    for i, (tag, over, f) in enumerate(jobs, 1):
        if deadline and (deadline - time.time()) < est:
            print(f"\n  {(deadline - time.time()) / 60:.0f} min left, a run "
                  f"needs about {est / 60:.0f} min. Stopping rather than "
                  f"starting one that would be killed mid-run.")
            break
        cmd = [sys.executable, "train_ib.py", "--cfg", CFG,
               "--data", DATA, "--split", split_of(f), "--tag", tag,
               "--set", "RESULT.OUTPUT_DIR", args.out,
               *BASE, *over, "SOLVER.SEED", SEED]
        print(f"\n{'=' * 68}\n  {tag}  fold {f}   ({i} of {len(jobs)})"
              f"   elapsed {_hms(time.time() - t0)}\n{'=' * 68}", flush=True)
        rc, dt = run_trial(cmd, f"{tag}/f{f}", i, len(jobs), MAX_EPOCH,
                           args.log, live, patience=PATIENCE)
        if rc != 0:
            print(f"  {tag} fold {f} FAILED (exit {rc}); continuing",
                  flush=True)
            continue
        durations.append(dt)
        est = max(durations)
        print(f"  {tag} fold {f} finished in {_hms(dt)}", flush=True)

    report(args.out)


if __name__ == "__main__":
    main()
