"""
KIBA queue, ordered so the most useful runs finish first.

KIBA is about four times the size of Davis (118,254 pairs against 30,056),
so a run costs roughly four times as much. Mirroring the full 40-run Davis
queue would be several hundred GPU-hours, which is not a sensible use of
one workstation. This queue is therefore deliberately smaller and ordered
by what the paper actually needs, so stopping early still leaves something
usable.

    python run_kiba.py --list
    python run_kiba.py --hours 8 --out ./results
    python run_kiba.py --only benchmark --hours 8 --out ./results

Groups, in the order they run
-----------------------------
benchmark  Canonical DeepDTA folds. The only KIBA numbers directly
           comparable with published tables, so these come first.
seeds      cold-target and scaffold at three seeds. Scaffold is the
           interesting one here: on Davis all 68 compounds have distinct
           scaffolds so the split degenerates to cold-drug, whereas KIBA
           has thousands of compounds across far fewer scaffolds, which
           makes it a real test of generalisation to new chemistry.
frontier   k=2, k=32 and the dense control at one seed. Enough to show
           whether the Davis finding, that sparsity costs nothing
           measurable, reproduces on a second dataset. Not enough to
           resolve a small effect, and not meant to.

Resumable in the same way as run_davis.py: a run with a readable
test_metrics.json is skipped, so re-running the command continues the
queue rather than restarting it.
"""
import argparse
import os
import subprocess
import sys
import time

# Share the resume, timing and reporting logic rather than copying it.
from run_davis import done_path, is_done

CFG = "configs/IBDTA.yaml"
DATA = "kiba"
SEEDS = [2048, 7, 13]


def _tag(prefix, seed):
    return prefix if seed == SEEDS[0] else f"{prefix}_seed{seed}"


def queue():
    q = []
    # Comparable-to-published first.
    for f in range(5):
        q.append(("benchmark", f"seed{SEEDS[0]}", f"benchmark/fold{f}",
                  ["SOLVER.SEED", str(SEEDS[0])]))
    # Then the two split regimes worth having on KIBA.
    for split in ("cold_target", "scaffold"):
        for s in SEEDS:
            q.append(("seeds", f"seed{s}", split, ["SOLVER.SEED", str(s)]))
    # Then a minimal frontier: the extreme budget, a middling one, and the
    # dense control.
    for k in (2, 32):
        q.append(("frontier", f"topk{k}", "cold_target",
                  ["IB.SELECTION", "topk", "IB.BUDGET_K", str(k),
                   "IB.ADAPTIVE_BUDGET", "false", "SOLVER.SEED", str(SEEDS[0])]))
    q.append(("frontier", "dense", "cold_target",
              ["IB.SELECTION", "softmax", "SOLVER.SEED", str(SEEDS[0])]))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0,
                    help="session budget; no run is started that is not "
                         "expected to finish within it")
    ap.add_argument("--est", type=float, default=8.0,
                    help="hours to assume a KIBA run takes until one has "
                         "been measured in this session")
    ap.add_argument("--out", default="./result")
    ap.add_argument("--only", nargs="*", default=None,
                    help="one or more of: benchmark seeds frontier")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    q = [r for r in queue() if not args.only or r[0] in args.only]

    if args.list:
        pend = 0
        for group, tag, split, _ in q:
            ok = is_done(args.out, split, tag, data=DATA)
            pend += 0 if ok else 1
            print(f"  [{'done' if ok else '    '}]  {group:9s} "
                  f"{split:18s} {tag}")
        print(f"\n  {len(q) - pend}/{len(q)} done, {pend} remaining "
              f"(~{pend * args.est:.0f} h at {args.est:.0f} h per run)")
        return

    if not os.path.isdir(os.path.join("datasets", DATA)):
        raise SystemExit(
            f"datasets/{DATA} missing. Run first:\n"
            f"  python download_data.py --datasets {DATA}\n"
            f"  python splits.py --csv datasets/{DATA}/full.csv "
            f"--out datasets/{DATA}")
    if any(r[2].startswith("benchmark/") for r in q) and not os.path.isdir(
            os.path.join("datasets", DATA, "benchmark")):
        print("  building canonical folds first")
        subprocess.run([sys.executable, "benchmark_split.py",
                        "--datasets", DATA], check=True)

    deadline = time.time() + args.hours * 3600
    est = args.est * 3600
    durations = []
    ran = skipped = 0
    for group, tag, split, over in q:
        if is_done(args.out, split, tag, data=DATA):
            skipped += 1
            continue
        left = deadline - time.time()
        if left <= 0:
            print(f"\n  session budget reached. {ran} run(s) this session.")
            break
        if left < est:
            print(f"\n  {left / 60:.0f} min left, a run needs about "
                  f"{est / 60:.0f} min. Stopping here rather than starting "
                  f"one that would be killed mid-run.")
            print(f"  {ran} run(s) this session. Resume with the same command.")
            break
        cmd = [sys.executable, "train_ib.py", "--cfg", CFG,
               "--data", DATA, "--split", split, "--tag", tag,
               "--set", "RESULT.OUTPUT_DIR", args.out, *over]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        print(f"\n{'=' * 68}\n  {group}  {DATA}/{split}  {tag}"
              f"   ({(deadline - time.time()) / 3600:.1f} h left)\n{'=' * 68}",
              flush=True)
        t0 = time.time()
        # Unbuffered child stdout, so progress reaches the log as it
        # happens rather than in 8 KB blocks.
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}); continuing", flush=True)
            continue
        ran += 1
        dt = time.time() - t0
        durations.append(dt)
        est = max(durations)
        print(f"  finished in {dt / 60:.1f} min "
              f"(estimate for the next run: {est / 60:.0f} min)", flush=True)
    else:
        print(f"\n  queue complete. {ran} run(s) this session, "
              f"{skipped} already done.")


if __name__ == "__main__":
    main()
