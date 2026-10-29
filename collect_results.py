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
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./result")
    ap.add_argument("--dest", default="results")
    args = ap.parse_args()

    df = load(args.out)
    if df.empty:
        raise SystemExit(f"no finished runs under {args.out}/davis/")
    os.makedirs(args.dest, exist_ok=True)

    keep = [c for c in ("run", "split", "selection", "mse", "rmse", "ci",
                        "r2", "pearson", "spearman", "support", "best_epoch")
            if c in df.columns]
    df = df[keep].sort_values("run")
    dest = os.path.join(args.dest, "davis_summary.csv")
    df.to_csv(dest, index=False)
    print(f"  {len(df)} run(s) -> {dest}\n")

    # Seed-averaged table, the one the headline numbers come from.
    seeded = df[df["run"].str.contains("seed")]
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

    # Budget sweep, for the frontier figure.
    front = df[df["run"].str.contains("topk|dense")]
    if not front.empty:
        p = os.path.join(args.dest, "frontier_measured.csv")
        front.to_csv(p, index=False)
        print(f"  wrote {p}")

    expected = 28
    if len(df) < expected:
        print(f"\n  {expected - len(df)} run(s) still missing. "
              f"`python run_davis.py --list --out {args.out}` shows which.")


if __name__ == "__main__":
    main()
