"""
The final Davis benchmark: five canonical folds at the tuned capacity.

The hyperparameter search found one lever that matters. Widening the
hidden dimension from 128 to 256 moved fold0 validation MSE from 0.3321
to 0.2899 and test MSE from 0.3540 to 0.3067, and the deletion audit on
that checkpoint came back clean (support 27.3 contacts, AFS +0.746,
exactness residual 9.1e-08). Nothing else in the sweep came close, and
the combination trials were abandoned because the machine running them
was taking fifty minutes an epoch.

This runs that configuration over all five canonical folds, with the test
evaluation and the deletion audit both enabled, so the number that goes
in the paper is measured the same way every other number in the paper is.

    python run_final.py                 # run the queue
    python run_final.py --list          # what is done, what is left
    python run_final.py --hours 11      # stop before a session cap

Resumable in the same way as the rest of the queue scripts: a fold with a
test_metrics.json is skipped, and a fold interrupted mid-training picks
up from its last.pt.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from keepawake import keep_awake
from run_tune import BASE, _enable_ansi, _hms, run_trial

CFG = "configs/IBDTA.yaml"
DATA = "davis"
TAG = "h256"
FOLDS = [0, 1, 2, 3, 4]
SEED = "2048"

# BASE lifts the epoch ceiling to 250 and adds patience 25, which is the
# regime every tuning trial was measured under. Running the final folds
# under the shipped 100-epoch ceiling would not reproduce the tuning
# result it is based on.
FINAL = BASE + ["IB.H_DIM", "256", "SOLVER.SEED", SEED]

MAX_EPOCH = int(BASE[BASE.index("SOLVER.MAX_EPOCH") + 1])
PATIENCE = int(BASE[BASE.index("SOLVER.PATIENCE") + 1])


def settings(patience=None, hdim=None):
    """Config overrides and the tag that records them.

    Patience and width both go into the tag. A fold measured under one
    stopping rule is not comparable with a fold measured under another,
    and a results directory that cannot say which rule produced it
    invites exactly that comparison.
    """
    over, tag = list(FINAL), TAG
    if hdim is not None:
        over[over.index("IB.H_DIM") + 1] = str(hdim)
        tag = f"h{hdim}"
    pat = PATIENCE
    if patience is not None:
        over[over.index("SOLVER.PATIENCE") + 1] = str(patience)
        tag, pat = f"{tag}_p{patience}", patience
    return over, tag, pat


def split_of(f):
    return f"benchmark/fold{f}"


def out_dir(out, f, tag=TAG):
    return os.path.join(out, DATA, f"{split_of(f)}_{tag}")


def is_done(out, f, tag=TAG):
    p = os.path.join(out_dir(out, f, tag), "test_metrics.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as fh:
            json.load(fh)
        return True
    except (ValueError, OSError):
        return False       # truncated by a killed session; redo it


def report(out, tag=TAG):
    import statistics
    vals = []
    print(f"\n  fold   test MSE    CI      support   AFS      epochs")
    print(f"  {'-' * 56}")
    for f in FOLDS:
        p = os.path.join(out_dir(out, f, tag), "test_metrics.json")
        if not os.path.exists(p):
            print(f"  {f}      pending")
            continue
        m = json.load(open(p))
        vals.append(m["mse"])
        print(f"  {f}      {m['mse']:.4f}      {m.get('ci', float('nan')):.3f}"
              f"   {m.get('support', float('nan')):7.1f}"
              f"   {m.get('afs', float('nan')):+.3f}"
              f"   {m.get('best_epoch', 0):>4}")
    if len(vals) >= 2:
        mu, sd = statistics.mean(vals), statistics.stdev(vals)
        print(f"  {'-' * 56}")
        print(f"  mean   {mu:.4f} +/- {sd:.4f}   over {len(vals)} fold(s)")
        print(f"\n  paper row:  ${mu:.3f}${{\\scriptsize$\\pm{sd:.3f}$}}")
    elif len(vals) == 1:
        print(f"\n  one fold only; no spread yet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./result")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="session budget in hours; 0 means no limit, which "
                         "is the right setting on a machine you own")
    ap.add_argument("--est", type=float, default=4.0,
                    help="hours to assume a fold takes until one has been "
                         "measured in this session")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--log", default="final_log.txt")
    ap.add_argument("--patience", type=int, default=None,
                    help="override early-stopping patience; runs are tagged "
                         "with it so one protocol never overwrites another")
    ap.add_argument("--hdim", type=int, default=None,
                    help="override the hidden width, if the sweep finds a "
                         "better one than 256")
    ap.add_argument("--no-live", action="store_true",
                    help="print one line every ten epochs instead of a "
                         "redrawing bar, for logs and schedulers")
    args = ap.parse_args()

    over, tag, patience = settings(args.patience, args.hdim)

    if args.list:
        report(args.out, tag)
        return

    _enable_ansi()
    # Before anything long starts. The queue process outlives every child,
    # so one request here covers all five folds.
    keep_awake()
    live = not args.no_live and sys.stdout.isatty()
    pend = [f for f in FOLDS if not is_done(args.out, f, tag)]
    print(f"  final benchmark, tag {tag}, patience {patience}, seed {SEED}")
    print(f"  {len(FOLDS) - len(pend)}/{len(FOLDS)} folds done, "
          f"{len(pend)} to run (~{len(pend) * args.est:.0f} h)")
    if not pend:
        report(args.out, tag)
        return

    deadline = time.time() + args.hours * 3600 if args.hours else None
    est = args.est * 3600
    durations = []
    t_start = time.time()
    for i, f in enumerate(pend, 1):
        if deadline:
            left = deadline - time.time()
            if left < est:
                print(f"\n  {left / 60:.0f} min left, a fold needs about "
                      f"{est / 60:.0f} min. Stopping rather than starting one "
                      f"that would be killed mid-run.")
                break
        cmd = [sys.executable, "train_ib.py", "--cfg", CFG,
               "--data", DATA, "--split", split_of(f), "--tag", tag,
               "--set", "RESULT.OUTPUT_DIR", args.out, *over]
        print(f"\n{'=' * 68}\n  fold {f}   ({i} of {len(pend)})"
              f"   elapsed {_hms(time.time() - t_start)}\n{'=' * 68}",
              flush=True)
        rc, dt = run_trial(cmd, f"fold{f}", i, len(pend), MAX_EPOCH,
                           args.log, live, patience=patience)
        if rc != 0:
            print(f"  fold {f} FAILED (exit {rc}); continuing", flush=True)
            continue
        durations.append(dt)
        est = max(durations)
        print(f"  fold {f} finished in {_hms(dt)}", flush=True)

    report(args.out, tag)


if __name__ == "__main__":
    main()
