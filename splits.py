"""
Split construction and leakage auditing.

Four regimes, all derived from an existing dataframe with SMILES / Protein /
label columns. No new data is collected.

    random        standard split. Reported for comparability only; no claim
                  in the paper rests on it.
    cold_drug     no test compound appears in training
    cold_target   no test protein appears in training
    scaffold      Bemis-Murcko scaffolds held out, so close analogues of a
                  training compound cannot appear in test

The leakage audit is not optional decoration. Reviewers in this area now
expect split integrity to be quantified, and the gap between random and
cold performance is itself a reportable result.

RDKit is required for scaffold splitting only. The other three regimes and
the audit run without it.
"""

import hashlib
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


SMILES_COL = "SMILES"
PROTEIN_COL = "Protein"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _hash(s):
    return hashlib.md5(str(s).encode()).hexdigest()


def murcko_scaffold(smiles, include_chirality=False):
    """Bemis-Murcko scaffold SMILES, or a hash of the molecule if unparseable."""
    if not _HAS_RDKIT:
        raise ImportError("scaffold splitting needs rdkit: pip install rdkit")
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality)
        return scaf if scaf else _hash(smiles)
    except Exception:
        return _hash(smiles)


def _grouped_split(df, groups, frac_train, frac_val, seed, balance_by_size=True):
    """Assign whole groups to splits. Largest groups first keeps sizes close
    to target when group sizes are very uneven (they are, for scaffolds)."""
    rng = np.random.RandomState(seed)
    by_group = defaultdict(list)
    for idx, g in zip(df.index, groups):
        by_group[g].append(idx)

    keys = list(by_group)
    rng.shuffle(keys)
    if balance_by_size:
        keys.sort(key=lambda k: -len(by_group[k]))

    n = len(df)
    targets = [frac_train * n, frac_val * n, (1 - frac_train - frac_val) * n]
    buckets = [[], [], []]
    sizes = [0, 0, 0]

    for k in keys:
        # fill whichever bucket is furthest below its target
        deficits = [t - s for t, s in zip(targets, sizes)]
        b = int(np.argmax(deficits))
        buckets[b].extend(by_group[k])
        sizes[b] += len(by_group[k])

    return [df.loc[sorted(b)].reset_index(drop=True) for b in buckets]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def make_split(df, regime, frac_train=0.7, frac_val=0.1, seed=42):
    """Return (train, val, test) dataframes for the given regime."""
    df = df.reset_index(drop=True)

    if regime == "random":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(df))
        n_tr = int(frac_train * len(df))
        n_va = int(frac_val * len(df))
        idx = [perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]]
        return [df.iloc[sorted(i)].reset_index(drop=True) for i in idx]

    if regime == "cold_drug":
        groups = df[SMILES_COL].values
    elif regime == "cold_target":
        groups = df[PROTEIN_COL].values
    elif regime == "scaffold":
        groups = np.array([murcko_scaffold(s) for s in df[SMILES_COL]])
    else:
        raise ValueError(f"unknown regime: {regime}")

    return _grouped_split(df, groups, frac_train, frac_val, seed)


def audit_leakage(train, val, test, scaffold=False):
    """Quantify overlap between splits.

    Report this table in the paper. A cold-target split with non-zero target
    overlap is not a cold-target split, and silent breakage here is the most
    common way published cold-start numbers turn out to be wrong.
    """
    def _ov(a, b, col):
        sa, sb = set(a[col]), set(b[col])
        return len(sa & sb), len(sb)

    rows = []
    for name, held in (("val", val), ("test", test)):
        d_shared, d_tot = _ov(train, held, SMILES_COL)
        p_shared, p_tot = _ov(train, held, PROTEIN_COL)
        row = {
            "split": name,
            "n": len(held),
            "drugs_seen_in_train": d_shared,
            "drugs_total": d_tot,
            "drug_overlap_pct": 100.0 * d_shared / max(d_tot, 1),
            "targets_seen_in_train": p_shared,
            "targets_total": p_tot,
            "target_overlap_pct": 100.0 * p_shared / max(p_tot, 1),
        }
        if scaffold:
            if not _HAS_RDKIT:
                warnings.warn("rdkit missing; skipping scaffold overlap")
            else:
                tr_s = {murcko_scaffold(s) for s in train[SMILES_COL]}
                he_s = [murcko_scaffold(s) for s in held[SMILES_COL]]
                shared = sum(1 for s in he_s if s in tr_s)
                row["scaffold_overlap_pct"] = 100.0 * shared / max(len(he_s), 1)
        # pair-level leakage: the same (drug, target) pair in both sets
        tr_pairs = set(zip(train[SMILES_COL], train[PROTEIN_COL]))
        he_pairs = list(zip(held[SMILES_COL], held[PROTEIN_COL]))
        row["pair_leakage_pct"] = 100.0 * sum(
            1 for p in he_pairs if p in tr_pairs) / max(len(he_pairs), 1)
        rows.append(row)

    return pd.DataFrame(rows)


def audit_duplicates(df, label_col="Y"):
    """Find (drug, protein-sequence) pairs that appear more than once.

    This matters more than it sounds. Davis lists 442 protein *entries* but
    only 379 unique *sequences*: kinase isoforms and mutants that share an
    identical sequence collapse onto the same key once you index by
    sequence text, which is what every sequence-based model actually sees.

    The result is pairs with identical model inputs and different labels.
    Measured on Davis: 1224 duplicate groups, 546 of them with disagreeing
    affinities, spread up to 4.7 pKd units. Two consequences:

      1. An irreducible error floor. No sequence-based model can get these
         right, because the inputs are identical. Reporting MSE without
         acknowledging it overstates how much headroom is left.
      2. Leakage in random splits. The duplicates land on both sides, so
         the model can memorise one copy and be tested on the other.

    Returns (summary_dict, duplicate_groups_dataframe).
    """
    g = df.groupby([SMILES_COL, PROTEIN_COL])[label_col].agg(
        n="count", n_unique="nunique", lo="min", hi="max")
    dup = g[g.n > 1].copy()
    dup["spread"] = dup.hi - dup.lo
    conflicting = dup[dup.n_unique > 1]

    # Error floor, weighted by how much of the dataset is actually affected.
    # For a group with identical inputs, the best any deterministic model
    # can do is predict the group mean, leaving that group's variance. The
    # dataset-wide floor is the row-weighted mean of those variances --
    # NOT the mean over conflicting groups alone, which would overstate it
    # by the inverse of their share.
    if len(conflicting):
        var = df.merge(conflicting.reset_index()[[SMILES_COL, PROTEIN_COL]],
                       on=[SMILES_COL, PROTEIN_COL])
        per_group_var = var.groupby([SMILES_COL, PROTEIN_COL])[label_col].var(ddof=0)
        n_rows = var.groupby([SMILES_COL, PROTEIN_COL])[label_col].size()
        floor = float((per_group_var * n_rows).sum() / len(df))
        affected = int(n_rows.sum())
    else:
        floor, affected = 0.0, 0

    summary = {
        "rows": len(df),
        "unique_pairs": len(g),
        "duplicate_groups": len(dup),
        "conflicting_groups": len(conflicting),
        "rows_in_conflict": affected,
        "pct_rows_affected": 100.0 * affected / max(len(df), 1),
        "max_spread": float(dup.spread.max()) if len(dup) else 0.0,
        "mean_conflict_spread": float(conflicting.spread.mean()) if len(conflicting) else 0.0,
        "dataset_mse_floor": floor,
    }
    return summary, dup.reset_index()


def deduplicate(df, how="mean", label_col="Y"):
    """Collapse duplicate (drug, sequence) pairs. how: mean | median | max | first."""
    agg = {"mean": "mean", "median": "median", "max": "max", "first": "first"}[how]
    out = (df.groupby([SMILES_COL, PROTEIN_COL], as_index=False)[label_col]
             .agg(agg))
    return out


def write_splits(df, out_dir, regime, **kw):
    """Materialise a split to train/val/test.csv under out_dir/regime/."""
    import os
    tr, va, te = make_split(df, regime, **kw)
    d = os.path.join(out_dir, regime)
    os.makedirs(d, exist_ok=True)
    tr.to_csv(os.path.join(d, "train.csv"), index=False)
    va.to_csv(os.path.join(d, "val.csv"), index=False)
    te.to_csv(os.path.join(d, "test.csv"), index=False)
    report = audit_leakage(tr, va, te, scaffold=(regime == "scaffold"))
    report.to_csv(os.path.join(d, "leakage_audit.csv"), index=False)
    return tr, va, te, report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build and audit DTA splits")
    ap.add_argument("--csv", required=True, help="full dataset csv")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--regimes", nargs="+",
                    default=["random", "cold_drug", "cold_target", "scaffold"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    full = pd.read_csv(args.csv)

    # Run the duplicate audit first: if the dataset contains contradictory
    # labels, every metric computed downstream has a floor that no model
    # can beat, and a random split will leak those duplicates across sides.
    print("=== data integrity ===")
    summary, dup = audit_duplicates(full)
    for k, v in summary.items():
        print(f"  {k:24s} {v:,.4f}" if isinstance(v, float)
              else f"  {k:24s} {v:,}")
    if summary["conflicting_groups"]:
        print(f"\n  {summary['conflicting_groups']:,} (drug, sequence) pairs carry "
              f"DIFFERENT labels.\n"
              f"  No sequence-based model can resolve them: the inputs are "
              f"identical.\n"
              f"  Implied MSE floor: {summary['dataset_mse_floor']:.4f}. "
              f"Use splits.deduplicate() to collapse them.")

    for r in args.regimes:
        tr, va, te, rep = write_splits(full, args.out, r, seed=args.seed)
        print(f"\n=== {r} ===  train {len(tr)}  val {len(va)}  test {len(te)}")
        print(rep.to_string(index=False))
