"""
Dataset downloader for IB-DTA.

Fetches everything the experiments need from public sources, converts it
into one uniform CSV schema, and reports what it got.

    SMILES, Protein, Y

Sources
-------
Davis, KIBA   DeepDTA repository (Ozturk et al. 2018, Bioinformatics)
              https://github.com/hkmztrk/DeepDTA
              Pickled dictionaries of ligands/proteins plus an affinity
              matrix. Davis is converted to pKd the way every paper in
              this line does (see `--no-log` to opt out).

BindingDB,    DrugBAN repository (Bai et al. 2023, Nat Mach Intell)
BioSNAP,      https://github.com/peizhenbai/DrugBAN
Human         Already split into train/val/test under random / cold /
              cluster regimes. Binary labels.

Usage
-----
    python download_data.py --all
    python download_data.py --datasets davis kiba
    python download_data.py --datasets bindingdb --subset 2000

Nothing here needs credentials. If your environment blocks raw.github-
usercontent.com the script says so plainly instead of writing empty files.
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

DEEPDTA = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"
DRUGBAN = "https://raw.githubusercontent.com/peizhenbai/DrugBAN/main/datasets"

DRUGBAN_SETS = {
    "bindingdb": ["random", "cluster"],
    "biosnap": ["random", "cluster"],
    "human": ["random", "cold"],
}
TIMEOUT = 60


def _get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "IB-DTA/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", errors="replace")


def _die_on_network(e, url):
    print(f"\n  FAILED: {url}\n  {type(e).__name__}: {e}", file=sys.stderr)
    print("\n  This environment cannot reach the dataset host. Options:\n"
          "    - run download_data.py where outbound HTTPS is allowed, then\n"
          "      copy the datasets/ folder across\n"
          "    - download the files by hand from the URLs above\n", file=sys.stderr)
    raise SystemExit(2)


# --------------------------------------------------------------------------
# Davis / KIBA  ->  regression CSVs
# --------------------------------------------------------------------------
def fetch_deepdta(name, out_root, to_log=True, subset=None):
    """Build a pKd/KIBA-score CSV from the DeepDTA release."""
    print(f"\n=== {name} (regression) ===")
    base = f"{DEEPDTA}/{name}"
    try:
        ligands = json.loads(_get(f"{base}/ligands_can.txt"))
        proteins = json.loads(_get(f"{base}/proteins.txt"))
        raw = _get(f"{base}/Y", binary=True)
    except (urllib.error.URLError, OSError) as e:
        _die_on_network(e, base)

    import pickle
    Y = pickle.loads(raw, encoding="latin1")
    Y = np.asarray(Y, dtype=float)

    smiles = list(ligands.values())
    seqs = list(proteins.values())
    print(f"  {len(smiles)} ligands x {len(seqs)} proteins -> matrix {Y.shape}")

    rows = []
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            v = Y[i, j]
            if np.isnan(v):
                continue
            rows.append((smiles[i], seqs[j], float(v)))
    df = pd.DataFrame(rows, columns=["SMILES", "Protein", "Y"])

    if name == "davis" and to_log:
        # Standard transform in the DTA literature: Kd (nM) -> pKd.
        # Davis is dense (no NaNs) and 10000 nM encodes "not measured /
        # weaker than the assay floor", which becomes pKd 5.0.
        df["Y"] = -np.log10(df["Y"] / 1e9)
        print(f"  converted Kd(nM) -> pKd   range {df.Y.min():.2f} .. {df.Y.max():.2f}")
    else:
        print(f"  score range {df.Y.min():.3f} .. {df.Y.max():.3f}")

    if subset:
        df = df.sample(min(subset, len(df)), random_state=0).reset_index(drop=True)
        print(f"  subsampled to {len(df)} rows")

    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "full.csv")
    df.to_csv(path, index=False)
    print(f"  wrote {len(df):,} pairs -> {path}")
    print(f"  {df.SMILES.nunique()} unique drugs, {df.Protein.nunique()} unique targets")
    return df


# --------------------------------------------------------------------------
# DrugBAN classification splits
# --------------------------------------------------------------------------
def fetch_drugban(name, out_root, splits=None, subset=None):
    print(f"\n=== {name} (classification, DrugBAN splits) ===")
    splits = splits or DRUGBAN_SETS[name]
    got = {}
    for sp in splits:
        d = os.path.join(out_root, name, sp)
        os.makedirs(d, exist_ok=True)
        # The `cluster` splits are DrugBAN's cross-domain setup and ship
        # under domain-adaptation filenames rather than train/val/test.
        parts = (("source_train", "target_train", "target_test")
                 if sp == "cluster" else ("train", "val", "test"))
        for part in parts:
            url = f"{DRUGBAN}/{name}/{sp}/{part}.csv"
            try:
                txt = _get(url)
            except (urllib.error.URLError, OSError) as e:
                if sp != splits[0]:
                    print(f"  skip {sp}/{part} ({e})")
                    continue
                _die_on_network(e, url)
            path = os.path.join(d, f"{part}.csv")
            with open(path, "w") as f:
                f.write(txt)
            n = len(txt.strip().splitlines()) - 1
            if subset and part == "train":
                df = pd.read_csv(path).sample(min(subset, n), random_state=0)
                df.to_csv(path, index=False)
                n = len(df)
            got[f"{sp}/{part}"] = n
            print(f"  {sp:10s} {part:5s} {n:7,} rows")
    return got


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Download DTA/DTI datasets into a uniform schema")
    ap.add_argument("--datasets", nargs="+",
                    default=None,
                    choices=["davis", "kiba", "bindingdb", "biosnap", "human"])
    ap.add_argument("--all", action="store_true", help="fetch everything")
    ap.add_argument("--out", default="./datasets")
    ap.add_argument("--subset", type=int, default=None,
                    help="cap rows (for quick tests / Colab)")
    ap.add_argument("--no-log", action="store_true",
                    help="keep Davis in raw Kd(nM) instead of pKd")
    args = ap.parse_args()

    names = (["davis", "kiba", "bindingdb", "biosnap", "human"]
             if args.all else (args.datasets or ["davis"]))
    os.makedirs(args.out, exist_ok=True)

    for n in names:
        if n in ("davis", "kiba"):
            fetch_deepdta(n, args.out, to_log=not args.no_log, subset=args.subset)
        else:
            fetch_drugban(n, args.out, subset=args.subset)

    print("\nDone. Next:")
    print(f"  python splits.py --csv {args.out}/davis/full.csv --out {args.out}/davis")
    print(f"  python train_ib.py --cfg configs/IBDTA.yaml --data davis --split cold_target")


if __name__ == "__main__":
    main()
