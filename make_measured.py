"""
Figures built from the measured 40-run Davis campaign.

Unlike make_illustrative.py, every value here is read from
results/davis_summary.csv. Nothing is invented, so nothing is watermarked.

    python make_measured.py --csv results/davis_summary.csv

Produces:
  fig3_frontier     the budget sweep against the dense control
  fig8_baselines    canonical-protocol comparison with published models
  fig9_coldstart    split regimes
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL1, COL2 = 3.45, 7.16

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "pdf.fonttype": 42, "figure.dpi": 150,
})

# Validated with the dataviz palette checker: all six checks pass against a
# light surface, worst adjacent CVD dE 18.9 (protan), normal-vision 27.3.
C_IB, C_REF = "#2d6ca8", "#c0392b"
INK, MUTED = "#222222", "#666666"

# Published values on the canonical DeepDTA protocol, quoted from the cited
# works as collated in MGraphDTA Table 3.
PUBLISHED = [("KronRLS", 0.379), ("SimBoost", 0.282), ("DeepDTA", 0.261),
             ("WideDTA", 0.262), ("GraphDTA", 0.229), ("MGraphDTA", 0.207)]


def save(fig, name, outdir):
    fig.savefig(os.path.join(outdir, f"{name}.pdf"),
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(outdir, f"{name}.png"),
                bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f"  wrote {outdir}/{name}.pdf  [measured]")


def fig_frontier(d, outdir):
    """The headline: every budget sits inside the dense model's own spread."""
    tk = d[d.selection == "topk"]
    de = d[d.selection == "softmax"]
    ks = np.array(sorted(tk.budget_k.unique()))
    mu = np.array([tk[tk.budget_k == k].mse.mean() for k in ks])
    sd = np.array([tk[tk.budget_k == k].mse.std() for k in ks])
    dmu, dsd = de.mse.mean(), de.mse.std()

    fig, ax = plt.subplots(figsize=(COL2, 3.3))

    # The dense model as a reference band rather than a line: the width IS
    # the argument, because every budget falls inside it.
    ax.axhspan(dmu - dsd, dmu + dsd, color=C_REF, alpha=0.13, zorder=0)
    ax.axhline(dmu, color=C_REF, lw=1.6, zorder=1)

    ax.errorbar(ks, mu, yerr=sd, fmt="o-", color=C_IB, capsize=2.5,
                elinewidth=1.0, capthick=1.0, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.set_xlabel("contacts retained, $k$   (dense retains 21,012)")
    ax.set_ylabel("test MSE  $\\downarrow$")
    ax.set_xlim(ks[0] * 0.80, ks[-1] * 1.62)
    lo = min((mu - sd).min(), dmu - dsd)
    hi = max((mu + sd).max(), dmu + dsd)
    pad = (hi - lo) * 0.16
    ax.set_ylim(lo - pad, hi + pad * 1.5)

    # Direct labels, placed clear of the data rather than over it.
    ax.text(ks[-1] * 1.07, dmu, f"dense\n{dmu:.3f}", color=C_REF,
            fontsize=7, va="center", ha="left", linespacing=1.3)
    ax.text(ks[0] * 0.95, dmu - dsd, "$\\pm$1 s.d.", color=C_REF,
            fontsize=6.3, va="bottom", ha="left", alpha=0.95)
    ax.text(ks[3], mu[3] + sd[3] + pad * 0.12, "top-$k$", color=C_IB,
            fontsize=7.5, va="bottom", ha="center")

    best = ks[int(np.argmin(mu))]
    ax.set_title("Every budget lies inside the dense model's own seed spread",
                 fontsize=8.6, pad=6)

    # k=2 earns a callout: it is the extreme case of the claim. Anchored
    # inside the axes so it cannot run into the title.
    ax.annotate(f"$k$=2: two contacts of 21,012,\n"
                f"10,506$\\times$ fewer, {mu[0] - dmu:+.3f} MSE",
                xy=(ks[0], mu[0] + sd[0]), xytext=(5.0, hi + pad * 0.55),
                fontsize=6.6, color=INK, ha="left", va="center",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=MUTED,
                                shrinkA=2, shrinkB=3))
    save(fig, "fig3_frontier", outdir)


def fig_baselines(d, outdir):
    """Canonical protocol against published models. Shows us behind."""
    b = d[d.run.str.startswith("benchmark")]
    ours, osd = b.mse.mean(), b.mse.std()
    names = [n for n, _ in PUBLISHED] + ["IB-DTA"]
    vals = [v for _, v in PUBLISHED] + [ours]
    cols = [MUTED] * len(PUBLISHED) + [C_IB]

    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    x = np.arange(len(names))
    ax.bar(x, vals, color=cols, width=0.62, zorder=2)
    ax.errorbar([x[-1]], [ours], yerr=[osd], fmt="none", ecolor=INK,
                capsize=3, elinewidth=1.0, zorder=3)
    for xi, v in zip(x, vals):
        off = osd + 0.014 if xi == x[-1] else 0.012   # clear the error bar
        ax.text(xi, v + off, f"{v:.3f}", ha="center", fontsize=6.4,
                color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=6.8)
    ax.set_ylabel("Davis MSE  $\\downarrow$")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_title("Canonical DeepDTA protocol: IB-DTA is behind "
                 "the published models", fontsize=8.2, pad=8)
    ax.text(0.5, -0.34, "Grey bars quoted from the cited works. "
            "IB-DTA is measured over the five folds, mean $\\pm$ s.d.",
            transform=ax.transAxes, ha="center", fontsize=6.2, color=MUTED)
    save(fig, "fig8_baselines", outdir)


def fig_coldstart(d, outdir):
    """Split regimes, ordered by difficulty."""
    base = d[(d.selection == "entmax") & (~d.run.str.startswith("benchmark"))]
    order = ["random", "cold_target", "cold_drug", "scaffold"]
    label = {"random": "random", "cold_target": "cold-target",
             "cold_drug": "cold-drug", "scaffold": "scaffold$^*$"}
    mu = [base[base.split == s].mse.mean() for s in order]
    sd = [base[base.split == s].mse.std() for s in order]
    n = [len(base[base.split == s]) for s in order]
    sd = [0.0 if np.isnan(v) else v for v in sd]

    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    x = np.arange(len(order))
    ax.bar(x, mu, yerr=sd, color=C_IB, width=0.6, capsize=3,
           error_kw=dict(elinewidth=1.0, ecolor=INK), zorder=2)
    for xi, v, s, ni in zip(x, mu, sd, n):
        ax.text(xi, v + s + 0.03, f"{v:.2f}", ha="center", fontsize=6.5,
                color=INK)
        ax.text(xi, 0.03, f"n={ni}", ha="center", fontsize=5.8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([label[s] for s in order], rotation=20, ha="right",
                       fontsize=6.8)
    ax.set_ylabel("Davis MSE  $\\downarrow$")
    ax.set_ylim(0, max(m + s for m, s in zip(mu, sd)) * 1.25)
    ax.set_title("Cost of holding chemistry out", fontsize=8.2, pad=6)
    save(fig, "fig9_coldstart", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/davis_summary.csv")
    ap.add_argument("--out", default="fig")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    os.makedirs(a.out, exist_ok=True)
    print(f"building measured figures from {len(d)} runs")
    fig_frontier(d, a.out)
    fig_baselines(d, a.out)
    fig_coldstart(d, a.out)
    print("done -- no watermarks, every value measured")
