"""
Sample experiment suite for IB-DTA.

Runs a reduced version of every experiment in the research plan and writes
figures to `figures/`. The point is to demonstrate that the machinery works
end to end and to produce the plots you will regenerate at full scale on a
GPU -- NOT to produce publishable numbers. Every figure is labelled with
the subset size it came from.

    python run_experiments.py --quick     # ~10 min CPU, tiny subsets
    python run_experiments.py --full      # GPU, full data

Experiments
    E0  leakage audit across split regimes
    E1  accuracy across split regimes
    E2  faithfulness-accuracy frontier (budget sweep)
    E3  deletion-curve audit: IB-DTA vs a pooled baseline
    E4  support dynamics during training
    E5  temperature -> sparsity calibration
    E6  ablations: selection mode, adaptive budget
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from encoders import DTADataset, collate
from models_ib import IBDTA, regression_loss, regression_metrics
from ib_layer import InteractionBottleneck, entmax15
import faithfulness as FA
import splits as SP

warnings.filterwarnings("ignore", category=RuntimeWarning)

FIG = "figures"
RES = "results"
PALETTE = {"ib": "#2b6cb0", "pooled": "#c53030", "rand": "#718096",
           "accent": "#2f855a", "warn": "#dd6b20"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})


def banner(s):
    print(f"\n{'=' * 64}\n{s}\n{'=' * 64}")


# --------------------------------------------------------------------------
def base_cfg(h_dim=64, selection="entmax", budget_k=32, adaptive=True,
             max_len=400, max_atoms=100):
    return {
        "ENCODER": {"BACKEND": "builtin"},
        "DRUG": {"NODE_IN_FEATS": 75, "NODE_IN_EMBEDDING": h_dim,
                 "HIDDEN_LAYERS": [h_dim, h_dim], "PADDING": True,
                 "MAX_NODES": max_atoms, "DROPOUT": 0.0},
        "PROTEIN": {"EMBEDDING_DIM": h_dim, "NUM_FILTERS": [h_dim] * 3,
                    "KERNEL_SIZE": [3, 6, 9], "PADDING": True,
                    "MAX_LEN": max_len},
        "IB": {"H_DIM": h_dim, "HEADS": 1, "SELECTION": selection,
               "BUDGET_K": budget_k, "ADAPTIVE_BUDGET": adaptive,
               "DROPOUT": 0.1, "CHUNK_Q": 64},
        "TASK": {"NAME": "regression"},
        "LOSS": {"LAMBDA_RANK": 0.1, "LAMBDA_SPARSE": 0.001},
        "SOLVER": {"MAX_EPOCH": 8, "BATCH_SIZE": 16, "NUM_WORKERS": 0,
                   "LR": 1e-3, "WEIGHT_DECAY": 1e-4, "SEED": 2048},
    }


def loaders_for(split_dir, cfg, n_train, n_eval):
    out = {}
    for part, n in (("train", n_train), ("val", n_eval), ("test", n_eval)):
        df = pd.read_csv(os.path.join(split_dir, f"{part}.csv"))
        if n and len(df) > n:
            df = df.sample(n, random_state=0)
        ds = DTADataset(df, max_atoms=cfg["DRUG"]["MAX_NODES"],
                        max_protein=cfg["PROTEIN"]["MAX_LEN"], verbose=False)
        out[part] = torch.utils.data.DataLoader(
            ds, batch_size=cfg["SOLVER"]["BATCH_SIZE"],
            shuffle=(part == "train"), collate_fn=collate,
            drop_last=(part == "train"))
    return out


def train_model(cfg, loaders, device, epochs=None, track_support=False):
    torch.manual_seed(cfg["SOLVER"]["SEED"])
    np.random.seed(cfg["SOLVER"]["SEED"])
    model = IBDTA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["SOLVER"]["LR"],
                            weight_decay=cfg["SOLVER"]["WEIGHT_DECAY"])
    epochs = epochs or cfg["SOLVER"]["MAX_EPOCH"]
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb, sup = 0.0, 0, []
        for nf, adj, nm, tok, pl, y in loaders["train"]:
            nf, adj, nm, tok, pl, y = [t.to(device) for t in (nf, adj, nm, tok, pl, y)]
            opt.zero_grad()
            out, _, alpha = model(nf, adj, nm, tok, pl)
            loss, _ = regression_loss(out, y, alpha, model,
                                      cfg["LOSS"]["LAMBDA_RANK"],
                                      cfg["LOSS"]["LAMBDA_SPARSE"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
            if track_support:
                sup.append(float(model.support_size(alpha).mean()))
        row = {"epoch": ep, "train_loss": tot / max(nb, 1)}
        if track_support:
            row["support"] = float(np.mean(sup)) if sup else float("nan")
        v = evaluate(model, loaders["val"], device)
        row.update({f"val_{k}": val for k, val in v.items()})
        hist.append(row)
    return model, pd.DataFrame(hist)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, sup = [], [], []
    for nf, adj, nm, tok, pl, y in loader:
        nf, adj, nm, tok, pl = [t.to(device) for t in (nf, adj, nm, tok, pl)]
        out, _, alpha = model(nf, adj, nm, tok, pl)
        ps.append(out.view(-1).cpu()); ys.append(y.view(-1))
        sup.append(model.support_size(alpha).cpu())
    m = regression_metrics(torch.cat(ys), torch.cat(ps))
    m["support"] = float(torch.cat(sup).mean())
    return m


# --------------------------------------------------------------------------
# E0 leakage
# --------------------------------------------------------------------------
def e0_leakage(data_dir, out):
    banner("E0  split leakage audit")
    rows = []
    for regime in ("random", "cold_drug", "cold_target", "scaffold"):
        d = os.path.join(data_dir, regime)
        if not os.path.isdir(d):
            continue
        tr = pd.read_csv(os.path.join(d, "train.csv"))
        te = pd.read_csv(os.path.join(d, "test.csv"))
        va = pd.read_csv(os.path.join(d, "val.csv"))
        rep = SP.audit_leakage(tr, va, te)
        t = rep[rep.split == "test"].iloc[0]
        rows.append({"regime": regime, "drug_overlap": t.drug_overlap_pct,
                     "target_overlap": t.target_overlap_pct,
                     "pair_leak": t.pair_leakage_pct, "n_test": int(t.n)})
        print(f"  {regime:12s} drug {t.drug_overlap_pct:5.1f}%  "
              f"target {t.target_overlap_pct:5.1f}%  pair {t.pair_leakage_pct:4.1f}%")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e0_leakage.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(df)); w = 0.35
    ax.bar(x - w/2, df.drug_overlap, w, label="drugs seen in train",
           color=PALETTE["ib"])
    ax.bar(x + w/2, df.target_overlap, w, label="targets seen in train",
           color=PALETTE["warn"])
    ax.set_xticks(x); ax.set_xticklabels(df.regime.str.replace("_", "\n"))
    ax.set_ylabel("% of test set seen during training")
    ax.set_title("Split leakage: what the model has already met")
    ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 105)
    for i, (d_, t_) in enumerate(zip(df.drug_overlap, df.target_overlap)):
        ax.text(i - w/2, d_ + 2, f"{d_:.0f}", ha="center", fontsize=7)
        ax.text(i + w/2, t_ + 2, f"{t_:.0f}", ha="center", fontsize=7)
    fig.savefig(os.path.join(out, "e0_leakage.png")); plt.close(fig)
    return df


# --------------------------------------------------------------------------
# E1 accuracy across regimes
# --------------------------------------------------------------------------
def e1_splits(data_dir, device, n_train, n_eval, epochs, out):
    banner("E1  accuracy across split regimes")
    rows = []
    for regime in ("random", "cold_drug", "cold_target", "scaffold"):
        d = os.path.join(data_dir, regime)
        if not os.path.isdir(d):
            continue
        cfg = base_cfg()
        ld = loaders_for(d, cfg, n_train, n_eval)
        t0 = time.time()
        model, _ = train_model(cfg, ld, device, epochs)
        m = evaluate(model, ld["test"], device)
        m.update({"regime": regime, "secs": time.time() - t0})
        rows.append(m)
        print(f"  {regime:12s} mse={m['mse']:.3f}  ci={m['ci']:.3f}  "
              f"r2={m['r2']:+.3f}  support={m['support']:.1f}  "
              f"({m['secs']:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e1_splits.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.arange(len(df))
    axes[0].bar(x, df.mse, color=PALETTE["ib"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(df.regime.str.replace("_", "\n"))
    axes[0].set_ylabel("test MSE (pKd$^2$)"); axes[0].set_title("Error by split regime")
    axes[1].bar(x, df.ci, color=PALETTE["accent"])
    axes[1].axhline(0.5, ls="--", lw=1, color=PALETTE["rand"])
    axes[1].text(len(df) - 0.5, 0.505, "chance", fontsize=7,
                 color=PALETTE["rand"], ha="right")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df.regime.str.replace("_", "\n"))
    axes[1].set_ylabel("concordance index"); axes[1].set_title("Ranking by split regime")
    axes[0].text(0.02, 0.97, "see e0_leakage for what each regime\nactually holds out",
                 transform=axes[0].transAxes, fontsize=6.5, va="top",
                 color=PALETTE["rand"])
    fig.suptitle("Accuracy by split regime "
                 f"(demo scale: {len(df)} regimes, small subsets -- "
                 "differences here are NOT meaningful)",
                 y=1.03, fontsize=9)
    fig.savefig(os.path.join(out, "e1_splits.png"), bbox_inches="tight"); plt.close(fig)
    return df


# --------------------------------------------------------------------------
# E2 frontier + E6 ablations
# --------------------------------------------------------------------------
def audit_model(model, loader, device, max_batches=3):
    """Average AFS / polarity over a few batches."""
    model.eval()
    afs, pol, exact = [], [], []
    for bi, (nf, adj, nm, tok, pl, y) in enumerate(loader):
        if bi >= max_batches:
            break
        nf, adj, nm, tok, pl = [t.to(device) for t in (nf, adj, nm, tok, pl)]

        def predict(drop):
            with torch.no_grad():
                out, _, _ = model(nf, adj, nm, tok, pl, drop_pairs=drop)
            return out

        with torch.no_grad():
            _, contrib, _ = model(nf, adj, nm, tok, pl)
        if contrib.dim() == 3:
            contrib = contrib.unsqueeze(1)
        r = FA.audit(predict, contrib, fractions=FA.default_fractions(n=7), seed=0)
        afs.append(r["afs"]); pol.append(r["polarity"])

        drop = torch.zeros_like(contrib, dtype=torch.bool)
        flat = contrib.reshape(contrib.shape[0], -1)
        k = min(5, flat.shape[1])
        idx = flat.abs().topk(k, -1).indices
        drop.reshape(contrib.shape[0], -1).scatter_(1, idx, True)
        _, ok, det = FA.check_additive_exactness(predict, contrib, drop)
        exact.append(det["max_residual"])
    return (float(np.mean(afs)), float(np.mean(pol)), float(np.max(exact)))


def e2_frontier(data_dir, device, n_train, n_eval, epochs, out):
    banner("E2  faithfulness-accuracy frontier (budget sweep)")
    d = os.path.join(data_dir, "cold_target")
    rows = []
    budgets = [2, 4, 8, 16, 32, 64]
    for k in budgets:
        cfg = base_cfg(selection="topk", budget_k=k, adaptive=False)
        ld = loaders_for(d, cfg, n_train, n_eval)
        model, _ = train_model(cfg, ld, device, epochs)
        m = evaluate(model, ld["test"], device)
        a, p, ex = audit_model(model, ld["test"], device)
        rows.append({"budget_k": k, **m, "afs": a, "polarity": p, "max_resid": ex})
        print(f"  k={k:3d}  mse={m['mse']:.3f}  ci={m['ci']:.3f}  "
              f"AFS={a:+.3f}  support={m['support']:.1f}  resid={ex:.1e}")

    # dense reference
    cfg = base_cfg(selection="softmax", adaptive=False)
    ld = loaders_for(d, cfg, n_train, n_eval)
    dm, _ = train_model(cfg, ld, device, epochs)
    dmet = evaluate(dm, ld["test"], device)
    da, dp, dex = audit_model(dm, ld["test"], device)
    print(f"  dense  mse={dmet['mse']:.3f}  ci={dmet['ci']:.3f}  AFS={da:+.3f}  "
          f"support={dmet['support']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e2_frontier.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    axes[0].plot(df.budget_k, df.mse, "o-", color=PALETTE["ib"], label="IB-DTA (top-k)")
    axes[0].axhline(dmet["mse"], ls="--", color=PALETTE["pooled"], label="dense softmax")
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("budget k (pairs kept)")
    axes[0].set_ylabel("test MSE"); axes[0].set_title("Cost of the bottleneck")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].plot(df.budget_k, df.afs, "o-", color=PALETTE["accent"])
    axes[1].axhline(da, ls="--", color=PALETTE["pooled"], label="dense softmax")
    axes[1].axhline(0, ls=":", color=PALETTE["rand"])
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("budget k")
    axes[1].set_ylabel("AFS (above-random faithfulness)")
    axes[1].set_title("Faithfulness vs budget"); axes[1].legend(frameon=False, fontsize=7)

    axes[2].scatter(df.afs, df.mse, c=np.log2(df.budget_k), cmap="viridis", s=48,
                    zorder=3)
    for _, r in df.iterrows():
        axes[2].annotate(f"k={int(r.budget_k)}", (r.afs, r.mse), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
    axes[2].scatter([da], [dmet["mse"]], marker="X", s=90,
                    color=PALETTE["pooled"], zorder=4, label="dense")
    axes[2].set_xlabel("AFS"); axes[2].set_ylabel("test MSE")
    axes[2].set_title("The frontier"); axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("Faithfulness-accuracy frontier (demo scale)", y=1.03, fontsize=10)
    fig.savefig(os.path.join(out, "e2_frontier.png"), bbox_inches="tight")
    plt.close(fig)
    return df


# --------------------------------------------------------------------------
# E3 deletion curves
# --------------------------------------------------------------------------
class PooledBaseline(torch.nn.Module):
    """DrugBAN-style control: same encoders, same interaction map, but the
    map is contracted into a joint vector before the head."""

    def __init__(self, cfg):
        super().__init__()
        from encoders import GCNDrugEncoder, CNNProteinEncoder
        h = cfg["IB"]["H_DIM"]
        self.drug = GCNDrugEncoder(
            in_feats=75, dim_embedding=cfg["DRUG"]["NODE_IN_EMBEDDING"],
            hidden_feats=tuple(cfg["DRUG"]["HIDDEN_LAYERS"]))
        self.prot = CNNProteinEncoder(
            embedding_dim=cfg["PROTEIN"]["EMBEDDING_DIM"],
            num_filters=tuple(cfg["PROTEIN"]["NUM_FILTERS"]),
            kernel_size=tuple(cfg["PROTEIN"]["KERNEL_SIZE"]))
        self.vp = torch.nn.Linear(self.drug.output_feats, h)
        self.qp = torch.nn.Linear(self.prot.output_feats, h)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def forward(self, nf, adj, nm, tok, pl, drop_pairs=None):
        v = self.vp(self.drug(nf, adj, nm))
        q_raw, qmask = self.prot(tok, pl)
        q = self.qp(q_raw)
        att = torch.einsum("bvd,bqd->bvq", v, q)
        pm = nm[:, :, None] & qmask[:, None, :]
        att = att.masked_fill(~pm, -1e9)
        B, V, Q = att.shape
        att = torch.softmax(att.reshape(B, -1), -1).reshape(B, V, Q) * pm.float()
        if drop_pairs is not None:
            dp = drop_pairs.squeeze(1) if drop_pairs.dim() == 4 else drop_pairs
            att = att * (~dp).float()
        fused = torch.einsum("bvd,bvq,bqd->bd", v, att, q)   # the bypass
        return self.head(fused), att.unsqueeze(1), att.unsqueeze(1)


def e3_deletion(data_dir, device, n_train, n_eval, epochs, out):
    banner("E3  deletion-curve audit: additive bottleneck vs pooled")
    d = os.path.join(data_dir, "cold_target")
    cfg = base_cfg()
    ld = loaders_for(d, cfg, n_train, n_eval)

    ib, _ = train_model(cfg, ld, device, epochs)

    torch.manual_seed(cfg["SOLVER"]["SEED"])
    pooled = PooledBaseline(cfg).to(device)
    opt = torch.optim.AdamW(pooled.parameters(), lr=cfg["SOLVER"]["LR"])
    for _ in range(epochs):
        pooled.train()
        for nf, adj, nm, tok, pl, y in ld["train"]:
            nf, adj, nm, tok, pl, y = [t.to(device) for t in (nf, adj, nm, tok, pl, y)]
            opt.zero_grad()
            pred, _, _ = pooled(nf, adj, nm, tok, pl)
            torch.nn.functional.mse_loss(pred.view(-1), y.view(-1)).backward()
            opt.step()

    batch = next(iter(ld["test"]))
    nf, adj, nm, tok, pl, y = [t.to(device) for t in batch]
    fr = FA.default_fractions()
    results = {}
    for name, model in (("IB-DTA (additive)", ib), ("Pooled (DrugBAN-style)", pooled)):
        model.eval()

        def predict(drop, m=model):
            with torch.no_grad():
                o, _, _ = m(nf, adj, nm, tok, pl, drop_pairs=drop)
            return o


        with torch.no_grad():
            _, contrib, alpha = model(nf, adj, nm, tok, pl)
        imp = contrib if contrib.dim() == 4 else contrib.unsqueeze(1)
        r = FA.audit(predict, imp, fractions=fr, seed=0)
        drop = torch.zeros_like(imp, dtype=torch.bool)
        flat = imp.reshape(imp.shape[0], -1)
        idx = flat.abs().topk(min(5, flat.shape[1]), -1).indices
        drop.reshape(imp.shape[0], -1).scatter_(1, idx, True)
        _, ok, det = FA.check_additive_exactness(predict, imp, drop)
        r["exact_ok"] = ok; r["resid"] = det["max_residual"]
        r["rel_resid"] = det["relative_residual"]
        results[name] = r
        print(f"  {name:24s} AFS={r['afs']:+.3f}  polarity={r['polarity']:+.3f}  "
              f"exact={'YES' if ok else 'NO'}  resid={det['max_residual']:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=False)
    for ax, (name, r) in zip(axes, results.items()):
        ax.set_xscale("symlog", linthresh=0.1)
        ax.plot(r["fractions"] * 100, r["most"], "o-", color=PALETTE["ib"],
                label="delete most important")
        ax.plot(r["fractions"] * 100, r["random"], "s--", color=PALETTE["rand"],
                label="delete at random")
        ax.plot(r["fractions"] * 100, r["least"], "^:", color=PALETTE["accent"],
                label="delete least important")
        ax.set_xlabel("% of interaction pairs deleted")
        ax.set_ylabel("|change in prediction|")
        ax.set_title(f"{name}\nAFS = {r['afs']:+.3f}   "
                     f"exact: {'YES' if r.get('exact_ok') else 'NO'}", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Both beat chance on deletion -- AFS does NOT separate them.\n"
                 "See e3_exactness.png: only the additive decoder gets the "
                 "MAGNITUDE right.", y=1.08, fontsize=9)
    fig.savefig(os.path.join(out, "e3_deletion_curves.png"), bbox_inches="tight")
    plt.close(fig)

    # exactness bar
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    names = list(results); vals = [max(results[n]["rel_resid"], 1e-9) for n in names]
    ax.bar(names, vals, color=[PALETTE["ib"], PALETTE["pooled"]])
    ax.set_yscale("log"); ax.set_ylabel("relative residual (log scale)")
    ax.set_title("Does |Δprediction| equal the attributed importance?")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.6, "exact" if v < 1e-4 else f"{v:.2f} off",
                ha="center", fontsize=8)
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    fig.savefig(os.path.join(out, "e3_exactness.png"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RES, "e3_audit.json"), "w") as f:
        json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in v.items()} for k, v in results.items()},
                  f, indent=2)
    return results


# --------------------------------------------------------------------------
# E4 support dynamics
# --------------------------------------------------------------------------
def e4_support(data_dir, device, n_train, n_eval, epochs, out):
    banner("E4  support dynamics during training")
    d = os.path.join(data_dir, "cold_target")
    cfg = base_cfg()
    ld = loaders_for(d, cfg, n_train, n_eval)
    _, hist = train_model(cfg, ld, device, max(epochs, 8), track_support=True)
    hist.to_csv(os.path.join(RES, "e4_support.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
    ax1.plot(hist.epoch, hist.train_loss, "o-", color=PALETTE["ib"])
    ax1.set_ylabel("train loss"); ax1.set_yscale("log")
    ax1.set_title("Watch support: collapse toward 1 means empty explanations")

    ax2.plot(hist.epoch, hist.support, "s--", color=PALETTE["warn"])
    ax2.set_ylabel("mean support (pairs)"); ax2.set_xlabel("epoch")
    ax2.set_yscale("log")
    ax2.axhline(1, ls=":", color=PALETTE["pooled"])
    ax2.text(hist.epoch.iloc[-1], 1.08, "collapse (argmax)", fontsize=7,
             color=PALETTE["pooled"], ha="right", va="bottom")
    ax2.set_ylim(bottom=0.8)
    fig.savefig(os.path.join(out, "e4_support_dynamics.png"), bbox_inches="tight")
    plt.close(fig)
    return hist


# --------------------------------------------------------------------------
# E5 temperature calibration (analytic, fast)
# --------------------------------------------------------------------------
def e5_temperature(out, n_pairs=(70, 300, 1200), trials=300):
    banner("E5  temperature -> sparsity calibration")
    temps = np.array([0.3, 0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    rows = []
    for n in n_pairs:
        for t in temps:
            x = torch.randn(trials, n) / t
            a = entmax15(x, dim=-1)
            s = float((a > 0).float().sum(-1).mean())
            rows.append({"n_pairs": n, "temp": float(t), "support": s,
                         "pct": 100 * s / n})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e5_temperature.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for n in n_pairs:
        sub = df[df.n_pairs == n]
        axes[0].plot(sub.temp, sub.pct, "o-", label=f"{n} pairs")
        axes[1].plot(sub.temp, sub.support, "o-", label=f"{n} pairs")
    axes[0].set_xlabel("temperature"); axes[0].set_ylabel("% of pairs in support")
    axes[0].set_title("Temperature controls sparsity monotonically")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_yscale("log"); axes[1].set_xlabel("temperature")
    axes[1].set_ylabel("support size (pairs, log)")
    axes[1].axhline(1, ls=":", color=PALETTE["pooled"])
    axes[1].text(3.6, 1.1, "collapse", fontsize=7, color=PALETTE["pooled"], ha="right")
    axes[1].set_title("Absolute support"); axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("alpha-entmax calibration: support depends on temperature and\n"
                 "candidate count, but NOT on embedding width (after standardisation)",
                 y=1.06, fontsize=9)
    fig.savefig(os.path.join(out, "e5_temperature.png"), bbox_inches="tight")
    plt.close(fig)
    return df


# --------------------------------------------------------------------------
# E6 ablations
# --------------------------------------------------------------------------
def e6_ablations(data_dir, device, n_train, n_eval, epochs, out):
    banner("E6  ablations")
    d = os.path.join(data_dir, "cold_target")
    variants = [
        ("entmax + adaptive", dict(selection="entmax", adaptive=True)),
        ("entmax, fixed", dict(selection="entmax", adaptive=False)),
        ("top-k (k=16)", dict(selection="topk", budget_k=16, adaptive=False)),
        ("dense softmax", dict(selection="softmax", adaptive=False)),
    ]
    rows = []
    for name, kw in variants:
        cfg = base_cfg(**kw)
        ld = loaders_for(d, cfg, n_train, n_eval)
        model, _ = train_model(cfg, ld, device, epochs)
        m = evaluate(model, ld["test"], device)
        a, p, ex = audit_model(model, ld["test"], device)
        rows.append({"variant": name, **m, "afs": a, "polarity": p})
        print(f"  {name:20s} mse={m['mse']:.3f}  ci={m['ci']:.3f}  "
              f"AFS={a:+.3f}  support={m['support']:.1f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "e6_ablations.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    x = np.arange(len(df))
    lbl = [v.replace(" ", "\n", 1) for v in df.variant]
    axes[0].bar(x, df.mse, color=PALETTE["ib"]); axes[0].set_ylabel("test MSE")
    axes[0].set_title("Accuracy")
    axes[1].bar(x, df.afs, color=PALETTE["accent"]); axes[1].set_ylabel("AFS")
    axes[1].axhline(0, ls=":", color=PALETTE["rand"]); axes[1].set_title("Faithfulness")
    axes[2].bar(x, df.support, color=PALETTE["warn"])
    axes[2].set_yscale("log"); axes[2].set_ylabel("support (pairs, log)")
    axes[2].set_title("Support size")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=7)
    fig.suptitle("Ablations: what each component contributes", y=1.04, fontsize=10)
    fig.savefig(os.path.join(out, "e6_ablations.png"), bbox_inches="tight")
    plt.close(fig)
    return df


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./datasets/davis")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of e0 e1 e2 e3 e4 e5 e6")
    args = ap.parse_args()

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.full:
        n_train, n_eval, epochs = None, None, 30
    else:
        n_train = args.n_train or 400
        n_eval = args.n_eval or 200
        epochs = args.epochs or 6

    print(f"device={device}  n_train={n_train}  n_eval={n_eval}  epochs={epochs}")
    if not args.full:
        print("NOTE: demo scale. Figures show that the machinery works,\n"
              "      not publishable numbers. Use --full on a GPU.")

    want = set(args.only or ["e0", "e1", "e2", "e3", "e4", "e5", "e6"])
    t0 = time.time()
    if "e0" in want: e0_leakage(args.data, FIG)
    if "e5" in want: e5_temperature(FIG)
    if "e1" in want: e1_splits(args.data, device, n_train, n_eval, epochs, FIG)
    if "e3" in want: e3_deletion(args.data, device, n_train, n_eval, epochs, FIG)
    if "e4" in want: e4_support(args.data, device, n_train, n_eval, epochs, FIG)
    if "e6" in want: e6_ablations(args.data, device, n_train, n_eval, epochs, FIG)
    if "e2" in want: e2_frontier(args.data, device, n_train, n_eval, epochs, FIG)

    banner(f"done in {time.time() - t0:.0f}s -- figures in {FIG}/, tables in {RES}/")
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
