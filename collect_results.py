"""
Turn the run_davis.py output into the paper's tables.

    python collect_results.py --out /kaggle/working/results

Reads every test_metrics.json under <out>/davis/, groups the runs, and
writes results/davis_summary.csv plus LaTeX fragments you can paste into
the manuscript. Runs that are missing are reported rather than silently
dropped, so a half-finished queue produces a half-finished table that says
which cells are still empty.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd


def load(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "davis", "*",
                                           "test_metrics.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except (ValueError, OSError):
            print(f"  unreadable, skipped: {p}")
            continue
        d["run"] = os.path.basename(os.path.dirname(p))
        rows.append(d)
    df = pd.DataFrame(rows)
    # A run interrupted and resumed, or written by an older version, can
    # leave a numeric field as a string. Coerce rather than crash on the
    # first groupby.
    for c in ("mse", "rmse", "ci", "r2", "pearson", "spearman", "support",
              "afs", "polarity", "relative_residual", "max_residual",
              "best_epoch", "budget_k", "params"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./result")
    ap.add_argument("--dest", default="results")
    args = ap.parse_args()

    df = load(args.out)
    if df.empty:
        raise SystemExit(f"no finished runs under {args.out}/davis/")
    os.makedirs(args.dest, exist_ok=True)

    keep = [c for c in ("run", "split", "selection", "budget_k", "adaptive",
                        "mse", "rmse", "ci", "r2", "pearson", "spearman",
                        "support", "afs", "polarity", "exact_ok",
                        "relative_residual", "best_epoch", "audit_error")
            if c in df.columns]
    df = df[keep].sort_values("run")
    dest = os.path.join(args.dest, "davis_summary.csv")
    df.to_csv(dest, index=False)
    print(f"  {len(df)} run(s) -> {dest}\n")

    # Seed-averaged table, the one the headline numbers come from.
    # Frontier runs are tagged topk8_seed7 and so on, so a bare "seed"
    # match pulls them into the split-regime table and inflates the
    # cold_target count with runs that are not the default configuration.
    seeded = df[df["run"].str.contains("seed")
                & ~df["run"].str.contains("topk|dense")]
    if not seeded.empty and "split" in seeded.columns:
        g = seeded.groupby("split")[["mse", "ci"]].agg(["mean", "std", "count"])
        print("  split regimes (mean +/- sd over seeds)")
        print(g.to_string(), "\n")
        lines = []
        def cell(mean, sd):
            # A single seed has no spread; printing \pm nan would be worse
            # than printing nothing.
            return (f"${mean:.3f}$" if pd.isna(sd)
                    else f"${mean:.3f}\\pm{sd:.3f}$")

        for split, r in g.iterrows():
            n = int(r[("mse", "count")])
            flag = "" if n >= 5 else f"  % only {n}/5 seeds"
            lines.append(
                f"{split.replace('_', '-')} & "
                f"{cell(r[('mse', 'mean')], r[('mse', 'std')])} & "
                f"{cell(r[('ci', 'mean')], r[('ci', 'std')])} \\\\{flag}")
        p = os.path.join(args.dest, "table_splits.tex")
        open(p, "w").write("\n".join(lines) + "\n")
        print(f"  wrote {p}")

    # Budget sweep, for the frontier figure. Averaged over seeds, because a
    # single-seed curve spans only a couple of standard deviations end to
    # end and its shape cannot be told apart from noise.
    front = df[df["run"].str.contains("topk|dense")]
    if not front.empty:
        p = os.path.join(args.dest, "frontier_measured.csv")
        front.to_csv(p, index=False)
        print(f"  wrote {p}")
        if "budget_k" in front.columns and "selection" in front.columns:
            fg = front.groupby(["selection", "budget_k"])[
                ["mse", "ci", "afs", "support"]].agg(["mean", "std", "count"])
            print("\n  frontier (mean +/- sd over seeds)")
            print(fg.to_string())
            thin = fg[("mse", "count")].min()
            if thin < 3:
                print(f"\n  WARNING: some budgets have only {int(thin)} seed(s). "
                      f"Seed-to-seed sd on this split is around 0.04 MSE, so "
                      f"the shape of the curve is not yet separable from noise.")

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_davis import queue
        expected = len(queue())
    except Exception:
        expected = None
    if expected and len(df) < expected:
        print(f"\n  {expected - len(df)} of {expected} run(s) still missing. "
              f"`python run_davis.py --list --out {args.out}` shows which.")


if __name__ == "__main__":
    main()
