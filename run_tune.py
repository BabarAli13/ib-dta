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
import re
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



# ----------------------------------------------------------------------
# Live progress
# ----------------------------------------------------------------------
EPOCH_RE = re.compile(
    r"epoch=(\d+)\s+train_loss=([\d.]+).*?secs=([\d.]+).*?val_mse=([\d.]+)"
    r".*?val_ci=([\d.]+)")


def _enable_ansi():
    """Windows consoles need VT processing turned on before carriage
    returns redraw cleanly. Harmless everywhere else."""
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass


def _hms(sec):
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _bar(frac, width=22):
    frac = min(max(frac, 0.0), 1.0)
    full = int(frac * width)
    return "#" * full + "." * (width - full)


def status_line(idx, total, tag, ep, max_epoch, vm, best, best_ep, vci,
                elapsed, per_ep, patience):
    """Progress against the expected stop, not the ceiling.

    With patience p, a run ends at the earlier of max_epoch and
    best_epoch + p. Drawing the bar against max_epoch would leave it
    nearly empty for a run that early-stops at epoch 60 of 250, which is
    the normal case and useless as feedback. The target moves outward
    whenever the model improves, which is itself informative: a bar that
    keeps receding means the run is still learning.
    """
    target = min(max_epoch, best_ep + patience) if patience else max_epoch
    target = max(target, ep)
    remaining = (target - ep) * per_ep
    return (f"[{idx}/{total} {tag:<11}] ep {ep:>3}/~{target:<3} "
            f"[{_bar(ep / max(target, 1))}] "
            f"val {vm:.4f} best {best:.4f}@{best_ep:<3} "
            f"CI {vci:.3f} {_hms(elapsed)}+~{_hms(remaining)}")


def run_trial(cmd, tag, idx, total, max_epoch, log_path, live,
              patience=0, env=None):
    """Run one trial, rendering progress as it goes.

    Child output is parsed rather than inherited so the screen can carry a
    single updating line. Every raw line still reaches log_path, so making
    the display readable costs nothing.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1", **(env or {}))
    t0 = time.time()
    best, best_ep, per_ep = float("inf"), 0, 0.0
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== trial {tag} =====\n")
        log.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = EPOCH_RE.search(line)
            if not m:
                s = line.rstrip()
                if s:
                    if live:
                        sys.stdout.write("\r" + " " * 118 + "\r")
                    print("   " + s[:112], flush=True)
                continue
            ep, per_ep = int(m.group(1)), float(m.group(3))
            vm, vci = float(m.group(4)), float(m.group(5))
            if vm < best:
                best, best_ep = vm, ep
            msg = status_line(idx, total, tag, ep, max_epoch, vm, best,
                              best_ep, vci, time.time() - t0, per_ep,
                              patience)
            if live:
                sys.stdout.write("\r" + msg[:118].ljust(118))
                sys.stdout.flush()
            elif ep % 10 == 0 or ep == 1:
                print(msg, flush=True)
        proc.wait()
    if live:
        sys.stdout.write("\r" + " " * 118 + "\r")
        sys.stdout.flush()
    return proc.returncode, time.time() - t0


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

    _enable_ansi()
    live = sys.stdout.isatty()
    os.makedirs(args.out, exist_ok=True)
    _base = dict(zip(BASE[::2], BASE[1::2]))
    max_epoch = int(_base["SOLVER.MAX_EPOCH"])
    patience = int(_base.get("SOLVER.PATIENCE", 0))
    started = time.time()
    deadline = started + args.hours * 3600
    est = args.est * 3600
    durations, ran = [], 0
    pending = [x for x in trials if not is_done(args.out, x[0])]
    print(f"\n  {len(trials) - len(pending)}/{len(trials)} trials already "
          f"done, {len(pending)} to run, budget {args.hours:.0f} h")
    print(f"  raw output -> {os.path.join(args.out, 'trials.log')}")
    if not live:
        print("  (output is redirected: printing a status line every "
              "10 epochs instead of a live bar)")
    print()
    for idx, (tag, over) in enumerate(trials, 1):
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
        print(f"  trial {idx}/{len(trials)}  {tag}   "
              f"{' '.join(over) or '(baseline)'}   "
              f"[{_hms(time.time() - started)} elapsed, "
              f"{left / 3600:.1f} h budget left]", flush=True)
        rc, dt = run_trial(cmd, tag, idx, len(trials), max_epoch,
                           os.path.join(args.out, "trials.log"), live,
                           patience)
        if rc != 0:
            print(f"  FAILED (exit {rc}); continuing", flush=True)
            continue
        ran += 1
        durations.append(dt)
        est = max(durations)
        v = val_of(args.out, tag)
        if v:
            print(f"  done {tag:<12} val MSE {v['val_mse']:.4f}  "
                  f"CI {v['val_ci']:.4f}  best epoch {v['best_epoch']}"
                  f"/{v['epochs_run']}  in {_hms(dt)}\n", flush=True)
    report(args.out)


if __name__ == "__main__":
    main()
