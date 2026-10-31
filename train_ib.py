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
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore last.pt and start the run from epoch 1")
    ap.add_argument("--no-audit", action="store_true",
                    help="skip the deletion audit after training")
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
    start_epoch = 1

    # Mid-run resume. A full-scale run is hours long, so losing one to a
    # power cut or a closed session is expensive. last.pt carries
    # everything needed to continue: weights, optimiser and scheduler
    # state, the best-so-far record and the history. It is removed once the
    # run finishes, so a completed run never looks resumable.
    last_path = os.path.join(out_dir, "last.pt")
    cfg_sig = json.dumps(cfg, sort_keys=True)
    if os.path.exists(last_path) and not args.no_resume:
        try:
            ck = torch.load(last_path, map_location=device, weights_only=False)
            if ck.get("cfg_sig") != cfg_sig:
                print("  last.pt is from a different configuration; "
                      "starting fresh")
            else:
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                sched.load_state_dict(ck["sched"])
                best = ck["best"]
                history = ck["history"]
                start_epoch = ck["epoch"] + 1
                print(f"  RESUMING from epoch {ck['epoch']} "
                      f"(best so far epoch {best['epoch']})")
        except Exception as e:
            print(f"  last.pt unreadable ({type(e).__name__}); starting fresh")

    for epoch in range(start_epoch, cfg["SOLVER"]["MAX_EPOCH"] + 1):
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
        # flush: when stdout is redirected to a file Python block-buffers
        # at about 8 KB, and an epoch line is ~160 bytes, so without this a
        # long run looks stalled for its first fifty epochs.
        print("  " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()), flush=True)

        if score > best["score"]:
            best = {"score": float(score), "epoch": epoch}
            if cfg["RESULT"]["SAVE_MODEL"]:
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

        # Write the resume point after every epoch, via a temporary file so
        # a crash during the write cannot leave a half-written checkpoint
        # where a readable one used to be.
        tmp = last_path + ".tmp"
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "history": history, "cfg_sig": cfg_sig},
                   tmp)
        os.replace(tmp, last_path)

        # Early stopping. The reported result is always the best checkpoint,
        # so stopping early can only change the answer if the run would have
        # improved after `patience` further epochs. On Davis cold_target the
        # best epoch was 17 of 50 and validation MSE rose monotonically after
        # roughly epoch 23, so the remaining epochs were pure cost.
        # PATIENCE: 0 restores the old behaviour of always running MAX_EPOCH.
        patience = int(cfg["SOLVER"].get("PATIENCE", 0))
        if patience and epoch - best["epoch"] >= patience:
            print(f"  early stop: no improvement for {patience} epochs "
                  f"(best was epoch {best['epoch']})")
            break

    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, loaders["test"], task, device)
    print("\nTEST (best epoch %d): " % best["epoch"] +
          "  ".join(f"{k}={v:.4f}" for k, v in test.items()))

    # Faithfulness audit on the best checkpoint. Done here, while the model
    # and loaders are live, because the frontier figure needs AFS against
    # accuracy and support for the same run, and recovering it afterwards
    # would mean rebuilding the model from a checkpoint and a config that
    # may not have been saved. Costs one batch of deletion curves.
    audit = {}
    if not args.no_audit:
        try:
            import faithfulness as FA
            batch = next(iter(loaders["test"]))
            nf, adj, nm, tok, pl = [
                b.to(device) if torch.is_tensor(b) else b for b in batch[:5]]

            def predict(drop):
                with torch.no_grad():
                    o = model(nf, adj, nm, tok, pl, drop_pairs=drop)
                return (o["y"] if isinstance(o, dict) else o[0]).squeeze(-1)

            with torch.no_grad():
                o = model(nf, adj, nm, tok, pl)
            contrib = o["contrib"] if isinstance(o, dict) else o[1]
            if contrib.dim() == 3:
                contrib = contrib.unsqueeze(1)
            r = FA.audit(predict, contrib, fractions=FA.default_fractions(n=9),
                         seed=0)
            drop = torch.zeros_like(contrib, dtype=torch.bool)
            flat = contrib.reshape(contrib.shape[0], -1)
            idx = flat.abs().topk(min(5, flat.shape[1]), -1).indices
            drop.reshape(contrib.shape[0], -1).scatter_(1, idx, True)
            _, ok, det = FA.check_additive_exactness(predict, contrib, drop)
            audit = {"afs": float(r["afs"]), "polarity": float(r["polarity"]),
                     "exact_ok": bool(ok),
                     "max_residual": float(det["max_residual"]),
                     "relative_residual": float(det["relative_residual"])}
            print(f"AUDIT: AFS={audit['afs']:+.4f}  "
                  f"polarity={audit['polarity']:+.4f}  "
                  f"exact={'YES' if ok else 'NO'}  "
                  f"resid={audit['max_residual']:.2e}")
        except Exception as e:
            # A failed audit must not cost the training run it followed.
            audit = {"audit_error": f"{type(e).__name__}: {e}"}
            print(f"AUDIT FAILED (training result is unaffected): {e}")

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump({**test, **audit, "best_epoch": best["epoch"],
                   "data": args.data,
                   "split": args.split, "selection": cfg["IB"]["SELECTION"],
                   "budget_k": cfg["IB"]["BUDGET_K"],
                   "adaptive": cfg["IB"]["ADAPTIVE_BUDGET"],
                   "params": n_par}, f, indent=2)
    # The run is complete, so the resume point is no longer meaningful and
    # would otherwise make a finished run look interruptible.
    if os.path.exists(last_path):
        try:
            os.remove(last_path)
        except OSError:
            pass
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
