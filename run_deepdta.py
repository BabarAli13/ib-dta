"""
DeepDTA retrained on our splits, as a second controlled baseline.

Table 1 quotes DeepDTA at 0.261 MSE from the published work. That number
was produced on the authors' own fold assignment, with their own training
budget and their own stopping rule, so the distance between it and ours
is not attributable to the models. The pooled-decoder control in
run_control.py isolates the decoder; this isolates the whole architecture
against a published reference point, by running the reference on the same
rows.

This is a reimplementation, not the authors' code, and the paper must say
so: a re-run row belongs beside the quoted one rather than replacing it,
and where the two disagree the disagreement is itself information about
how much of a published table is protocol.

Faithful to Ozturk et al. 2018 (Bioinformatics 34(17):i821): character
embeddings of SMILES and sequence at width 128, three 1-D convolutions
per branch with 32/64/96 filters, kernel 4/6/8 for the ligand and 4/8/12
for the protein, global max pooling, then 1024/1024/512 fully connected
with dropout 0.1. No graph library, no pretrained weights, so it runs on
a free Kaggle GPU in well under an hour a fold and can proceed in
parallel with the lab machine.

    python run_deepdta.py --root datasets/davis --out ./result
    python run_deepdta.py --list

Writes test_metrics.json in the same shape every other runner here uses,
so collect_results.py picks the rows up without special-casing. The
attribution columns are written as null rather than zero: DeepDTA max-pools
its features and has no interaction map at all, so a support or an AFS for
it would be a fabrication, not a measurement.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models_ib import regression_metrics

# Vocabularies from the DeepDTA reference implementation. Kept verbatim
# rather than derived from the data, so an unseen character maps to 0 here
# exactly as it would there.
CHARCANSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33,
                 ".": 2, "1": 34, "0": 3, "3": 35, "2": 4, "5": 36,
                 "4": 5, "7": 37, "6": 6, "9": 38, "8": 7, "=": 39,
                 "A": 40, "@": 8, "C": 42, "B": 9, "E": 43, "D": 10,
                 "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47,
                 "L": 13, "O": 48, "N": 14, "P": 15, "S": 50, "R": 49,
                 "U": 51, "T": 16, "W": 52, "V": 17, "Y": 54, "[": 55,
                 "Z": 18, "]": 56, "\\": 19, "a": 57, "c": 58, "b": 20,
                 "e": 59, "d": 21, "g": 60, "f": 22, "i": 61, "h": 23,
                 "m": 62, "l": 24, "o": 63, "n": 25, "s": 64, "r": 26,
                 "u": 65, "t": 27, "y": 66, "p": 28}
CHARPROTSET = {"A": 1, "C": 3, "B": 2, "E": 5, "D": 4, "G": 7, "F": 6,
               "I": 9, "H": 8, "K": 11, "M": 13, "L": 12, "O": 15,
               "N": 14, "Q": 17, "P": 16, "S": 19, "R": 18, "U": 21,
               "T": 20, "W": 23, "V": 22, "Y": 25, "X": 24, "Z": 26}
# Size the embeddings from the vocabularies, not from the published
# CHARCANSMILEN of 64. The reference set has 64 entries but assigns
# indices up to 66, with 41 and 53 unused, so a hard 64 would raise an
# index error on any ligand containing 'u' or 'y' and on no other input.
N_SMI = max(CHARCANSMISET.values())
N_PROT = max(CHARPROTSET.values())
MAX_SMI, MAX_PROT = 100, 1000


def label(seq, vocab, maxlen):
    x = np.zeros(maxlen, dtype=np.int64)
    for i, ch in enumerate(seq[:maxlen]):
        x[i] = vocab.get(ch, 0)
    return x


class DeepDTASet(Dataset):
    def __init__(self, df):
        self.d = [label(s, CHARCANSMISET, MAX_SMI) for s in df.SMILES]
        self.p = [label(s, CHARPROTSET, MAX_PROT) for s in df.Protein]
        self.y = df.Y.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.d[i]), torch.from_numpy(self.p[i]),
                torch.tensor(self.y[i]))


class DeepDTA(nn.Module):
    def __init__(self, emb=128, filters=32, drop=0.1):
        super().__init__()
        self.de = nn.Embedding(N_SMI + 1, emb, padding_idx=0)
        self.pe = nn.Embedding(N_PROT + 1, emb, padding_idx=0)
        self.dc = nn.Sequential(
            nn.Conv1d(emb, filters, 4), nn.ReLU(),
            nn.Conv1d(filters, filters * 2, 6), nn.ReLU(),
            nn.Conv1d(filters * 2, filters * 3, 8), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))
        self.pc = nn.Sequential(
            nn.Conv1d(emb, filters, 4), nn.ReLU(),
            nn.Conv1d(filters, filters * 2, 8), nn.ReLU(),
            nn.Conv1d(filters * 2, filters * 3, 12), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))
        self.head = nn.Sequential(
            nn.Linear(filters * 6, 1024), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, d, p):
        dv = self.dc(self.de(d).transpose(1, 2)).squeeze(-1)
        pv = self.pc(self.pe(p).transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([dv, pv], 1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for d, p, y in loader:
        ps.append(model(d.to(device), p.to(device)).cpu())
        ys.append(y)
    return regression_metrics(torch.cat(ys), torch.cat(ps))


def one_fold(root, split, out, tag, args, device):
    folder = os.path.join(root, split)
    tr = pd.read_csv(os.path.join(folder, "train.csv"))
    va = pd.read_csv(os.path.join(folder, "val.csv"))
    # A benchmark fold holds train and val only: all five folds share one
    # independent test set a level up. Same lookup train_ib.py uses.
    te_path = os.path.join(folder, "test.csv")
    if not os.path.exists(te_path):
        te_path = os.path.join(os.path.dirname(folder.rstrip("/\\")),
                               "test.csv")
    te = pd.read_csv(te_path)

    mk = lambda df, sh: DataLoader(DeepDTASet(df), batch_size=args.batch,
                                   shuffle=sh, drop_last=sh, num_workers=2)
    ltr, lva, lte = mk(tr, True), mk(va, False), mk(te, False)
    print(f"  {split}  {len(tr)}/{len(va)}/{len(te)} rows", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = DeepDTA().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    run_dir = os.path.join(out, "davis", f"{split}_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    best = {"val_mse": math.inf, "epoch": 0}
    hist, t0 = [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = n = 0
        for d, p, y in ltr:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(
                model(d.to(device), p.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        m = evaluate(model, lva, device)
        hist.append({"epoch": ep, "train_loss": tot / n,
                     **{f"val_{k}": v for k, v in m.items()}})
        if m["mse"] < best["val_mse"]:
            best = {"val_mse": m["mse"], "epoch": ep}
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch={ep:4d}  train={tot / n:.4f}  "
                  f"val_mse={m['mse']:.4f}  val_ci={m['ci']:.4f}  "
                  f"best={best['val_mse']:.4f}@{best['epoch']}  "
                  f"{(time.time() - t0) / 60:.0f}m", flush=True)
        if ep - best["epoch"] >= args.patience:
            print(f"  early stop at {ep}, no gain for {args.patience} epochs",
                  flush=True)
            break

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"),
                                     map_location=device))
    test = evaluate(model, lte, device)
    pd.DataFrame(hist).to_csv(os.path.join(run_dir, "history.csv"),
                              index=False)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        # support/afs/polarity are null, not zero. DeepDTA max-pools its
        # features and has no interaction map, so any value here would be
        # invented rather than measured.
        json.dump({**test, "best_epoch": best["epoch"],
                   "data": "davis", "split": split, "model": "DeepDTA",
                   "reimplementation": True,
                   "support": None, "afs": None, "polarity": None,
                   "exact_ok": None, "relative_residual": None}, f, indent=2)
    print(f"  TEST mse={test['mse']:.4f} ci={test['ci']:.4f} "
          f"(best val epoch {best['epoch']})\n", flush=True)
    return test["mse"]


def report(out, tag):
    vals = []
    print(f"\n  fold   test MSE      CI    epochs")
    print(f"  {'-' * 38}")
    for f in range(5):
        p = os.path.join(out, "davis", f"benchmark/fold{f}_{tag}",
                         "test_metrics.json")
        if not os.path.exists(p):
            print(f"  {f}      pending")
            continue
        m = json.load(open(p))
        vals.append(m["mse"])
        print(f"  {f}      {m['mse']:.4f}    {m['ci']:.3f}"
              f"    {m['best_epoch']:>4}")
    if len(vals) > 1:
        import statistics as s
        print(f"  {'-' * 38}")
        print(f"  mean   {s.mean(vals):.4f} +/- {s.stdev(vals):.4f} "
              f"over {len(vals)} folds")
        print(f"\n  published DeepDTA on the authors' own folds: 0.261")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/davis")
    ap.add_argument("--out", default="./result")
    ap.add_argument("--tag", default="deepdta")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2048)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        report(args.out, args.tag)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  DeepDTA reimplementation on our splits, device={device}")
    print(f"  epochs<={args.epochs}  patience={args.patience}  "
          f"seed={args.seed}")
    for f in args.folds:
        split = f"benchmark/fold{f}"
        done = os.path.join(args.out, "davis", f"{split}_{args.tag}",
                            "test_metrics.json")
        if os.path.exists(done):
            print(f"  fold {f} already done, skipping")
            continue
        print(f"\n{'=' * 60}\n  fold {f}\n{'=' * 60}", flush=True)
        one_fold(args.root, split, args.out, args.tag, args, device)
    report(args.out, args.tag)


if __name__ == "__main__":
    main()
