"""
From attribution to design: counterfactual evaluation on matched
molecular pairs.

Motivation
----------
Exact attribution is only worth having if it supports a decision. The
decision a medicinal chemist actually faces is: *which part of this
molecule should I change, and what will that do to affinity?*

An additive interaction bottleneck answers it directly. Because

    y_hat = b + sum_{(i,j) in S} c_ij                                (1)

the attribution of a single atom is the exact marginal

    phi_i = sum_j c_ij                                               (2)

and removing that atom's contacts moves the prediction by exactly phi_i
(Prop. 8). For a set of atoms the effects compose additively with no
interaction correction (Prop. 9). No pooled model can make either
statement, because its contributions do not decompose.

The evaluation
--------------
Matched molecular pairs (MMPs) give ground truth for this. Two compounds
sharing a scaffold but differing in one substituent, both measured against
the same target, provide a *measured* affinity difference:

    delta_true = y(A, P) - y(B, P)

The model predicts the same quantity from attribution alone, by summing
the marginals of the atoms that differ:

    delta_pred = sum_{i in A \\ B} phi_i^A  -  sum_{i in B \\ A} phi_i^B

If attribution is meaningful, these correlate. Critically this is a
*harder* test than the deletion audit: deletion asks whether the model is
self-consistent, which Prop. 1 guarantees trivially. This asks whether the
attribution tracks physical reality, which nothing guarantees.

KIBA supplies 5,290 matched compound pairs over 281 shared scaffolds and
184,134 (pair, target) triples with both affinities measured. Davis
supplies none, because all 68 of its compounds have distinct scaffolds ---
the same degeneracy reported for its scaffold split.

No new data is collected; MMPs are derived from the existing files.
"""

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFMCS
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False

SMILES_COL, PROTEIN_COL, LABEL = "SMILES", "Protein", "Y"


# --------------------------------------------------------------------------
# MMP extraction
# --------------------------------------------------------------------------
def find_mmps(df, min_delta=0.0, max_pairs=None, mcs_timeout=2,
              min_core_frac=0.6, verbose=True):
    """Extract matched molecular pairs sharing a target.

    Two compounds form an MMP when they share a Bemis--Murcko scaffold and
    their maximum common substructure covers at least `min_core_frac` of
    both. The differing atoms are the substituent being swapped.

    Returns a dataframe with one row per (compound A, compound B, target):
        smiles_a, smiles_b, protein, y_a, y_b, delta,
        diff_atoms_a, diff_atoms_b   (atom index lists)
    """
    if not _HAS_RDKIT:
        raise ImportError("design.py needs rdkit: pip install rdkit")

    smis = df[SMILES_COL].unique()
    mols = {}
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols[s] = m

    by_scaffold = defaultdict(list)
    for s, m in mols.items():
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            continue
        if sc:
            by_scaffold[sc].append(s)

    # affinity lookup: protein -> smiles -> y
    idx = defaultdict(dict)
    for r in df.itertuples(index=False):
        idx[getattr(r, PROTEIN_COL)][getattr(r, SMILES_COL)] = getattr(r, LABEL)

    rows, n_scanned = [], 0
    for scaf, members in by_scaffold.items():
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(members, 2):
            n_scanned += 1
            if max_pairs and len(rows) >= max_pairs:
                break
            ma, mb = mols[a], mols[b]
            try:
                mcs = rdFMCS.FindMCS([ma, mb], timeout=mcs_timeout,
                                     ringMatchesRingOnly=True,
                                     completeRingsOnly=True)
            except Exception:
                continue
            if mcs.canceled or mcs.numAtoms == 0:
                continue
            core_a = mcs.numAtoms / ma.GetNumAtoms()
            core_b = mcs.numAtoms / mb.GetNumAtoms()
            if core_a < min_core_frac or core_b < min_core_frac:
                continue

            patt = Chem.MolFromSmarts(mcs.smartsString)
            if patt is None:
                continue
            match_a = ma.GetSubstructMatch(patt)
            match_b = mb.GetSubstructMatch(patt)
            if not match_a or not match_b:
                continue
            diff_a = sorted(set(range(ma.GetNumAtoms())) - set(match_a))
            diff_b = sorted(set(range(mb.GetNumAtoms())) - set(match_b))
            if not diff_a and not diff_b:
                continue

            for prot, d in idx.items():
                if a in d and b in d:
                    delta = float(d[a] - d[b])
                    if abs(delta) < min_delta:
                        continue
                    rows.append({
                        "smiles_a": a, "smiles_b": b, "protein": prot,
                        "y_a": float(d[a]), "y_b": float(d[b]),
                        "delta": delta,
                        "n_core": mcs.numAtoms,
                        "diff_atoms_a": diff_a, "diff_atoms_b": diff_b,
                    })
    out = pd.DataFrame(rows)
    if verbose:
        print(f"  scanned {n_scanned:,} same-scaffold compound pairs")
        print(f"  kept {len(out):,} (pair, target) triples "
              f"with both affinities measured")
        if len(out):
            print(f"  |delta| mean {out.delta.abs().mean():.3f}, "
                  f"median {out.delta.abs().median():.3f}, "
                  f"max {out.delta.abs().max():.3f}")
    return out


# --------------------------------------------------------------------------
# Counterfactual prediction from attribution
# --------------------------------------------------------------------------
def atom_attribution(contrib):
    """Atom-level marginal phi_i = sum_j c_ij.   contrib: (B,H,V,Q) -> (B,V)"""
    if contrib.dim() == 3:
        contrib = contrib.unsqueeze(1)
    return contrib.sum(dim=(1, 3))


def residue_attribution(contrib):
    """Residue-level marginal.  contrib: (B,H,V,Q) -> (B,Q)"""
    if contrib.dim() == 3:
        contrib = contrib.unsqueeze(1)
    return contrib.sum(dim=(1, 2))


def predict_swap(phi_a, diff_a, phi_b, diff_b):
    """Predicted affinity change from swapping substituents.

    phi_a, phi_b : (V_a,), (V_b,) atom attributions for the two compounds
    diff_a, diff_b : atom indices unique to each

    Returns the predicted delta = y_a - y_b implied by attribution alone.
    """
    ga = float(sum(phi_a[i] for i in diff_a if i < len(phi_a)))
    gb = float(sum(phi_b[i] for i in diff_b if i < len(phi_b)))
    return ga - gb


def evaluate_counterfactuals(pred_deltas, true_deltas):
    """Correlation and sign agreement between predicted and measured change.

    Sign accuracy is the decision-relevant number: a chemist asking
    "will this substitution help?" needs the direction right, and a model
    that is well correlated but systematically wrong in sign is useless.
    """
    p = np.asarray(pred_deltas, float)
    t = np.asarray(true_deltas, float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if len(p) < 2:
        return {"n": int(len(p))}
    from scipy.stats import pearsonr, spearmanr
    nz = np.abs(t) > 1e-9
    return {
        "n": int(len(p)),
        "pearson": float(pearsonr(p, t)[0]),
        "spearman": float(spearmanr(p, t).statistic),
        "sign_acc": float((np.sign(p[nz]) == np.sign(t[nz])).mean()),
        "mae": float(np.abs(p - t).mean()),
        "rmse": float(np.sqrt(((p - t) ** 2).mean())),
    }


def summarise(df):
    """Dataset-level MMP statistics for reporting."""
    if not len(df):
        return {"n_triples": 0}
    return {
        "n_triples": len(df),
        "n_compound_pairs": len(df.groupby(["smiles_a", "smiles_b"])),
        "n_targets": df.protein.nunique(),
        "mean_abs_delta": float(df.delta.abs().mean()),
        "median_abs_delta": float(df.delta.abs().median()),
        "max_abs_delta": float(df.delta.abs().max()),
        "frac_delta_gt_0.5": float((df.delta.abs() > 0.5).mean()),
        "frac_delta_gt_1.0": float((df.delta.abs() > 1.0).mean()),
    }


if __name__ == "__main__":
    import argparse, json, os
    ap = argparse.ArgumentParser(
        description="Extract matched molecular pairs for counterfactual eval")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/mmp.csv")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--min-delta", type=float, default=0.0)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    print(f"loaded {len(df):,} rows, {df[SMILES_COL].nunique():,} compounds")
    mmp = find_mmps(df, min_delta=a.min_delta, max_pairs=a.max_pairs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    mmp.to_csv(a.out, index=False)
    s = summarise(mmp)
    print(json.dumps(s, indent=2))
    with open(a.out.replace(".csv", "_summary.json"), "w") as f:
        json.dump(s, f, indent=2)
