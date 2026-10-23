"""
Training entry point for IB-DTA.

Base model: DrugBAN (Bai et al., Nature Machine Intelligence 2023).
IB-DTA keeps its encoders and replaces the bilinear-pooling interaction
module with a sparse additive bottleneck.

Examples
--------
# regression on Davis, cold-target split
python train_ib.py --cfg configs/IBDTA.yaml --data davis --split cold_target

# classification on the splits DrugBAN ships
python train_ib.py --cfg configs/IBDTA_cls.yaml --data bindingdb --split random

# sweep the budget for the faithfulness-accuracy frontier
for k in 4 8 16 32 64; do
  python train_ib.py --cfg configs/IBDTA.yaml --data davis --split scaffold \
      --set IB.SELECTION topk IB.BUDGET_K $k --tag k$k
done
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from encoders import DTADataset, collate
from models_ib import (IBDTA, regression_loss, classification_loss,
                       regression_metrics, classification_metrics)


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_cfg(path, overrides=None):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        node, leaf = k.rsplit(".", 1)
        d = cfg
        for part in node.split("."):
            d = d.setdefault(part, {})
        # keep the yaml type
        old = d.get(leaf)
        if isinstance(old, bool):
            v = str(v).lower() in ("1", "true", "yes")
        elif isinstance(old, int) and not isinstance(old, bool):
            v = int(v)
        elif isinstance(old, float):
            v = float(v)
        d[leaf] = v
    return cfg


def make_loader(df, cfg, shuffle, verbose=True):
    ds = DTADataset(df, max_atoms=cfg["DRUG"]["MAX_NODES"],
                    max_protein=cfg["PROTEIN"]["MAX_LEN"], verbose=verbose)
    return DataLoader(ds, batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                      shuffle=shuffle, num_workers=cfg["SOLVER"]["NUM_WORKERS"],
                      collate_fn=collate, drop_last=shuffle)


@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    ys, ps, sup = [], [], []
    for nf, adj, nm, tok, pl, y in loader:
        nf, adj, nm, tok, pl = [t.to(device) for t in (nf, adj, nm, tok, pl)]
        out, _, alpha = model(nf, adj, nm, tok, pl)
        ps.append(out.view(-1).cpu())
        ys.append(y.view(-1))
        sup.append(model.support_size(alpha).cpu())
    y_true, y_pred = torch.cat(ys), torch.cat(ps)
    m = (regression_metrics(y_true, y_pred) if task == "regression"
         else classification_metrics(y_true, torch.sigmoid(y_pred)))
    m["support"] = float(torch.cat(sup).mean())
    return m


def main():
    ap = argparse.ArgumentParser(description="IB-DTA training")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="random")
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None,
                    help="cap training rows (quick runs)")
    ap.add_argument("--eval-subset", type=int, default=None,
                    help="cap val/test rows (quick runs)")
    ap.add_argument("--set", nargs="*", default=None,
                    help="cfg overrides: --set IB.BUDGET_K 16 IB.SELECTION topk")
    args = ap.parse_args()

    overrides = {}
    if args.set:
        if len(args.set) % 2:
            raise SystemExit("--set takes KEY VALUE pairs")
        overrides = dict(zip(args.set[::2], args.set[1::2]))

    cfg = load_cfg(args.cfg, overrides)
    if args.epochs:
        cfg["SOLVER"]["MAX_EPOCH"] = args.epochs
    task = cfg["TASK"]["NAME"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["SOLVER"]["SEED"])

    folder = os.path.join(cfg.get("DATA_ROOT", "./datasets"), args.data, args.split)
    if not os.path.isdir(folder):
        raise SystemExit(
            f"{folder} not found.\n"
            f"  python download_data.py --datasets {args.data}\n"
            f"  python splits.py --csv ./datasets/{args.data}/full.csv "
            f"--out ./datasets/{args.data}")

    dfs = {}
    for s in ("train", "val", "test"):
        p = os.path.join(folder, f"{s}.csv")
        if not os.path.exists(p):
            # The canonical benchmark protocol shares ONE independent test
            # set across all five CV folds, so fold directories hold only
            # train.csv and val.csv. Fall back to the parent directory.
            alt = os.path.join(os.path.dirname(folder.rstrip("/")), f"{s}.csv")
            if s == "test" and os.path.exists(alt):
                p = alt
            else:
                raise SystemExit(f"missing {p}")
        dfs[s] = pd.read_csv(p)
    if args.subset:
        dfs["train"] = dfs["train"].sample(
            min(args.subset, len(dfs["train"])), random_state=0)
    if args.eval_subset:
        for s_ in ("val", "test"):
            dfs[s_] = dfs[s_].sample(
                min(args.eval_subset, len(dfs[s_])), random_state=0)

    print(f"device={device}  task={task}  selection={cfg['IB']['SELECTION']}  "
          f"budget_k={cfg['IB']['BUDGET_K']}  adaptive={cfg['IB']['ADAPTIVE_BUDGET']}")
    print(f"data={args.data}/{args.split}  "
          f"{len(dfs['train'])}/{len(dfs['val'])}/{len(dfs['test'])} rows")

    loaders = {s: make_loader(d, cfg, shuffle=(s == "train"))
               for s, d in dfs.items()}

    model = IBDTA(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_par:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["SOLVER"]["LR"],
                            weight_decay=cfg["SOLVER"].get("WEIGHT_DECAY", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["SOLVER"]["MAX_EPOCH"])

    tag = args.split + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(cfg["RESULT"]["OUTPUT_DIR"], args.data, tag)
    os.makedirs(out_dir, exist_ok=True)

    best = {"score": -np.inf, "epoch": -1}
    history = []
    for epoch in range(1, cfg["SOLVER"]["MAX_EPOCH"] + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for nf, adj, nm, tok, pl, y in loaders["train"]:
            nf, adj, nm, tok, pl, y = [t.to(device)
                                       for t in (nf, adj, nm, tok, pl, y)]
            opt.zero_grad()
            out, _, alpha = model(nf, adj, nm, tok, pl)
            if task == "regression":
                loss, _ = regression_loss(
                    out, y, alpha, model,
                    cfg["LOSS"]["LAMBDA_RANK"], cfg["LOSS"]["LAMBDA_SPARSE"])
            else:
                _, loss, _ = classification_loss(
                    out, y, alpha, model, cfg["LOSS"]["LAMBDA_SPARSE"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()

        val = evaluate(model, loaders["val"], task, device)
        score = -val["mse"] if task == "regression" else val["auroc"]
        row = {"epoch": epoch, "train_loss": tot / max(nb, 1),
               "secs": time.time() - t0,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print("  " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()))

        if score > best["score"]:
            best = {"score": float(score), "epoch": epoch}
            if cfg["RESULT"]["SAVE_MODEL"]:
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, loaders["test"], task, device)
    print("\nTEST (best epoch %d): " % best["epoch"] +
          "  ".join(f"{k}={v:.4f}" for k, v in test.items()))

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump({**test, "best_epoch": best["epoch"], "data": args.data,
                   "split": args.split, "selection": cfg["IB"]["SELECTION"],
                   "budget_k": cfg["IB"]["BUDGET_K"],
                   "adaptive": cfg["IB"]["ADAPTIVE_BUDGET"],
                   "params": n_par}, f, indent=2)
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
