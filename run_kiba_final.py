"""
KIBA: the Davis result, asked again of a second dataset.

Davis now carries three claims that depend on measurement rather than
proof: the model is $0.078$ MSE behind a pooled decoder, that difference
is the decode and not the encoders because the pooled arm on shared
encoders matches a retrained DeepDTA, and the attribution survives at a
support of tens. A single dataset cannot establish any of them. This runs
the two arms that matter on KIBA's canonical folds.

The old run_kiba.py predates the patience protocol, keep-awake and the
pooled decoder, and it ordered the queue by what the paper needed in
October. This replaces it for the benchmark folds; the split regimes and
frontier groups there are still useful and are left alone.

**Arms are interleaved by fold**, additive then pooled on fold 0, then
both on fold 1, and so on. The obvious order, one arm then the other,
looks tidier and is wrong: KIBA is four times the size of Davis and the
additive arm costs roughly a day a fold, so a queue stopped halfway would
leave five of one arm and none of the other, which answers no question.
Interleaved, stopping at any point leaves n matched folds, which is
exactly what the paired test consumes.

    python run_kiba_final.py --list
    python run_kiba_final.py                 # the queue
    python run_kiba_final.py --only pooled   # the cheap arm alone

The pooled arm is about ten times faster per epoch, since it contracts
the map once rather than scoring every selected contact, so if the
machine is unreliable it is the arm that will finish.
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
DATA = "kiba"
FOLDS = [0, 1, 2, 3, 4]
SEED = "2048"

MAX_EPOCH = int(BASE[BASE.index("SOLVER.MAX_EPOCH") + 1])
PATIENCE = int(BASE[BASE.index("SOLVER.PATIENCE") + 1])

# Same two decoders as the Davis controlled comparison, at the capacity
# the Davis sweep settled on. Changing the width here as well would mean
# the KIBA numbers answered a different question from the Davis ones.
ARMS = [
    ("h256", ["IB.DECODER", "additive", "IB.H_DIM", "256"]),
    ("pooled128", ["IB.DECODER", "pooled", "IB.H_DIM", "128"]),
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
        return False


def rows(out, tag):
    r = []
    for f in FOLDS:
        p = os.path.join(out_dir(out, f, tag), "test_metrics.json")
        if os.path.exists(p):
            try:
                r.append((f, json.load(open(p))))
            except (ValueError, OSError):
                pass
    return r


def report(out):
    import math
    seen = {}
    for tag, _ in ARMS:
        r = rows(out, tag)
        seen[tag] = {f: m for f, m in r}
        print(f"\n  {tag}")
        if not r:
            print("    pending")
            continue
        print(f"    {'fold':>5}{'MSE':>10}{'CI':>9}{'support':>10}"
              f"{'AFS':>9}{'epochs':>8}")
        for f, m in r:
            sup = m.get("support")
            afs = m.get("afs")
            print(f"    {f:>5}{m['mse']:>10.4f}{m['ci']:>9.3f}"
                  f"{(f'{sup:.1f}' if sup is not None else '-'):>10}"
                  f"{(f'{afs:+.3f}' if afs is not None else '-'):>9}"
                  f"{m.get('best_epoch', 0):>8}")
        v = [m["mse"] for _, m in r]
        if len(v) > 1:
            print(f"    mean {statistics.mean(v):.4f} "
                  f"+/- {statistics.stdev(v):.4f}  over {len(v)} folds")

    # The paired comparison, over whatever folds both arms have reached.
    a, b = ARMS[0][0], ARMS[1][0]
    common = sorted(set(seen.get(a, {})) & set(seen.get(b, {})))
    print()
    if len(common) < 2:
        print(f"  paired {a} - {b}: needs 2 matched folds, have "
              f"{len(common)}")
        return
    d = [seen[a][f]["mse"] - seen[b][f]["mse"] for f in common]
    m, sd = statistics.mean(d), statistics.stdev(d)
    se = sd / math.sqrt(len(d))
    # Two-sided 95% t critical values, df 1..9. Hard-coded so the report
    # needs no scipy, which is not installed on every machine this runs on.
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    t = tcrit.get(len(d) - 1, 1.96)
    print(f"  paired {a} - {b} over {len(common)} folds: {m:+.4f}  "
          f"95% CI [{m - t * se:+.4f}, {m + t * se:+.4f}]  "
          f"{sum(1 for x in d if x > 0)}/{len(d)} folds")
    print(f"  Davis gave +0.0775 [+0.036, +0.119] on five folds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./result")
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--est", type=float, default=12.0,
                    help="hours to assume a run takes before one is "
                         "measured. KIBA is about four times Davis, and "
                         "the two arms differ tenfold, so this is only a "
                         "first guess; it is replaced per arm as soon as "
                         "that arm has finished one run.")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--log", default="kiba_log.txt")
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()

    if args.list:
        report(args.out)
        return

    _enable_ansi()
    keep_awake()
    live = not args.no_live and sys.stdout.isatty()

    arms = [a for a in ARMS if not args.only or a[0] in args.only]
    jobs = [(tag, over, f) for f in FOLDS for tag, over in arms
            if not is_done(args.out, f, tag)]
    total = len(arms) * len(FOLDS)
    print(f"  KIBA canonical folds, seed {SEED}")
    print(f"  {total - len(jobs)}/{total} runs done, {len(jobs)} to go")
    if not jobs:
        report(args.out)
        return

    deadline = time.time() + args.hours * 3600 if args.hours else None
    # Per arm, because a tenfold difference between them makes a single
    # shared estimate useless for deciding whether a run will fit.
    est = {tag: args.est * 3600 for tag, _ in ARMS}
    t0 = time.time()
    for i, (tag, over, f) in enumerate(jobs, 1):
        if deadline and (deadline - time.time()) < est[tag]:
            print(f"\n  {(deadline - time.time()) / 60:.0f} min left, a "
                  f"{tag} run needs about {est[tag] / 60:.0f} min. "
                  f"Stopping rather than starting one that would be "
                  f"killed mid-run.")
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
        est[tag] = dt
        print(f"  {tag} fold {f} finished in {_hms(dt)}", flush=True)

    report(args.out)


if __name__ == "__main__":
    main()
