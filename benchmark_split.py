"""
The canonical Davis/KIBA benchmark protocol.

Why this file exists
--------------------
Retraining every baseline on your own splits gives a *controlled*
comparison, which is the scientifically correct way to attribute a
difference to an architecture. But it leaves a reviewer unable to place
your numbers against the published literature, because your splits are
not theirs.

Both comparisons are needed, and they answer different questions:

  standard protocol  -> "are these numbers in the right ballpark?"
                        Comparable with every published Davis/KIBA result.
  controlled splits  -> "is the difference attributable to the method?"
                        Comparable only within your own study.

This module provides the first. It loads the exact fold assignment
distributed with DeepDTA (Oztuk et al., Bioinformatics 2018), which is the
protocol used by GraphDTA, MGraphDTA, AttentionDTA and most of the
subsequent literature:

  - the dataset is divided into six equal parts
  - one part is held out as an independent test set
  - the remaining five are used for 5-fold cross-validation
  - the reported score is the mean over the five models on that same
    held-out test set

For Davis this gives a test set of 5,010 pairs and five training folds of
5,009--5,010 each.

A caution worth stating in any paper that uses these numbers: the
literature is NOT uniform. MGraphDTA and GraphDTA follow the DeepDTA
folds, but NHGNN-DTA (Bioinformatics 2023) uses an 8:1:1 TDC split
instead. Quoting both in one table without noting the difference compares
models evaluated under different conditions. `describe_protocol()` prints
the caveat so it is hard to forget.

Source of the fold files:
  https://github.com/hkmztrk/DeepDTA/tree/master/data/{davis,kiba}/folds
"""

import json
import os
import urllib.request

import numpy as np
import pandas as pd

DEEPDTA_FOLDS = ("https://raw.githubusercontent.com/hkmztrk/DeepDTA/"
                 "master/data/{ds}/folds/{f}")
TIMEOUT = 60


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "IB-DTA/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def download_folds(dataset, out_dir):
    """Fetch train/test fold indices for 'davis' or 'kiba'."""
    d = os.path.join(out_dir, dataset, "folds")
    os.makedirs(d, exist_ok=True)
    out = {}
    for name, key in (("train_fold_setting1.txt", "train"),
                      ("test_fold_setting1.txt", "test")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as f:
                out[key] = json.load(f)
        else:
            out[key] = _fetch(DEEPDTA_FOLDS.format(ds=dataset, f=name))
            with open(p, "w") as f:
                json.dump(out[key], f)
    return out["train"], out["test"]


def _dense_index(dataset, data_root):
    """Rebuild the ligand-major flat index the fold files refer to.

    The fold indices address the affinity matrix flattened in row-major
    order over (ligand, protein), counting only non-NaN entries in the
    order they appear. We reconstruct that ordering from the same source
    `download_data.py` uses, so the mapping is exact rather than assumed.
    """
    import pickle
    base = f"https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/{dataset}"
    ligands = json.loads(_fetch_text(f"{base}/ligands_can.txt"))
    proteins = json.loads(_fetch_text(f"{base}/proteins.txt"))
    raw = _fetch_bytes(f"{base}/Y")
    Y = np.asarray(pickle.loads(raw, encoding="latin1"), dtype=float)

    smiles = list(ligands.values())
    seqs = list(proteins.values())
    rows = []
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if np.isnan(Y[i, j]):
                continue
            rows.append((smiles[i], seqs[j], float(Y[i, j])))
    return pd.DataFrame(rows, columns=["SMILES", "Protein", "Y"])


def _fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "IB-DTA/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", errors="replace")


def _fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "IB-DTA/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read()


def build_benchmark_splits(dataset, data_root="./datasets", to_log=True,
                           verbose=True):
    """Materialise the canonical protocol as CSV files.

    Writes, under <data_root>/<dataset>/benchmark/:
        fold0..fold4/{train,val}.csv    the 5 CV folds
        test.csv                        the shared independent test set

    Returns a summary dict.
    """
    train_folds, test_idx = download_folds(dataset, data_root)
    df = _dense_index(dataset, data_root)

    if dataset == "davis" and to_log:
        df["Y"] = -np.log10(df["Y"] / 1e9)

    n = len(df)
    flat_train = [i for f in train_folds for i in f]
    covered = set(flat_train) | set(test_idx)
    if max(covered) >= n:
        raise RuntimeError(
            f"fold indices exceed dataset size ({max(covered)} >= {n}); "
            "the affinity matrix ordering does not match the fold files")

    out = os.path.join(data_root, dataset, "benchmark")
    os.makedirs(out, exist_ok=True)
    df.iloc[sorted(test_idx)].to_csv(os.path.join(out, "test.csv"), index=False)

    for k, val_fold in enumerate(train_folds):
        tr = [i for j, f in enumerate(train_folds) if j != k for i in f]
        d = os.path.join(out, f"fold{k}")
        os.makedirs(d, exist_ok=True)
        df.iloc[sorted(tr)].to_csv(os.path.join(d, "train.csv"), index=False)
        df.iloc[sorted(val_fold)].to_csv(os.path.join(d, "val.csv"), index=False)

    summary = {
        "dataset": dataset,
        "total_pairs": n,
        "test_size": len(test_idx),
        "n_folds": len(train_folds),
        "fold_sizes": [len(f) for f in train_folds],
        "train_per_fold": len(flat_train) - len(train_folds[0]),
        "coverage": len(covered) / n,
    }
    if verbose:
        print(f"  {dataset}: {n:,} pairs -> test {len(test_idx):,}, "
              f"{len(train_folds)} folds of {summary['fold_sizes']}")
        print(f"  index coverage: {100*summary['coverage']:.1f}%")
        print(f"  written to {out}")
    return summary


def describe_protocol():
    print(__doc__)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Build the canonical DeepDTA benchmark splits")
    ap.add_argument("--datasets", nargs="+", default=["davis"],
                    choices=["davis", "kiba"])
    ap.add_argument("--out", default="./datasets")
    ap.add_argument("--describe", action="store_true")
    a = ap.parse_args()
    if a.describe:
        describe_protocol()
    for d in a.datasets:
        build_benchmark_splits(d, a.out)
