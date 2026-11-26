"""
Do the selected contacts land on the ATP pocket?

Every faithfulness result in this repository is internal: Prop. 1 says
the attribution describes the model, and the deletion audit confirms it
numerically. Neither says the selected contacts are the ones a
crystallographer would pick. That is the obvious objection to an
interpretability paper in a bioinformatics venue, and this script is the
answer to it.

Both benchmarks are kinase panels, so the ATP pocket is locatable from
sequence by the catalytic motifs (kinase_motifs.py). The test is whether
the contacts the model selects concentrate there above chance, and
whether they do so more than a pooled attention map's largest weights at
the same support.

    python validate_biology.py --audit --data davis
    python validate_biology.py --run result/davis/benchmark/fold0_h256
    python validate_biology.py --run <additive> --control <pooled>

Three things this script is careful about, because each would otherwise
produce a flattering number.

Resolution. The protein encoder is three valid convolutions with kernels
3, 6 and 9, so output index j is a function of input residues j..j+15.
The model cannot localise to a residue, only to a 16-residue window, and
a selected contact is scored as a hit when its window overlaps the
pocket. Scoring the window centre instead would understate; requiring the
whole window inside the pocket would overstate. Any claim from this
script is about windows.

Matched support. The additive model selects tens of contacts and the
pooled control assigns weight to all 21,012. Comparing "the pooled map's
pocket fraction" against "the additive model's pocket fraction" would
compare a dense distribution with a sparse one. The control therefore
takes the pooled map's top-N weights, N being the additive model's
support on that same complex.

Chance level. Pocket occupancy differs per protein, so a single global
baseline is wrong. Enrichment is computed per complex against that
protein's own pocket fraction, then aggregated.
"""
import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch

import kinase_motifs as KM
from models_ib import IBDTA
from train_ib import make_loader

CROP = 15          # (3-1)+(6-1)+(9-1); see CNNProteinEncoder.out_len
RF = CROP + 1      # receptive field of one output position, in residues


def _lgamma_choose(n, k):
    if k < 0 or k > n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1)
            - math.lgamma(n - k + 1))


def hypergeom_sf(k, M, n, N):
    """P(X >= k) for X ~ Hypergeometric(M population, n successes, N draws).

    Written out rather than imported so the script runs without scipy,
    which is not installed on every machine this has to run on.
    """
    if k <= 0:
        return 1.0
    total = _lgamma_choose(M, N)
    acc = 0.0
    for i in range(k, min(n, N) + 1):
        acc += math.exp(_lgamma_choose(n, i)
                        + _lgamma_choose(M - n, N - i) - total)
    return min(1.0, max(0.0, acc))


def pocket_output_indices(seq, n_out, criterion="overlap", strict=False):
    """Output positions counted as being in the ATP pocket.

    Two criteria, because the choice sets the chance level and therefore
    the enrichment, and neither is obviously right.

    "overlap": the window touches the pocket anywhere. Faithful to what
    the model can resolve, but a 71-residue pocket covering 21% of a
    sequence covers 44% of 16-residue windows, which caps enrichment at
    about 2.3x. Permissive.

    "center": the window's central residue is in the pocket. Chance falls
    back to roughly the pocket's residue fraction and the ceiling rises,
    at the cost of scoring a window that genuinely clips the pocket as a
    miss. Strict.

    We report both. Quoting only the one that flatters the result would be
    the same error as quoting a single deletion fraction.

    Returns (indices, pocket_residue_count), or (None, None) when the
    kinase domain could not be resolved.
    """
    pocket = KM.pocket_residues(seq, strict=strict)
    if pocket is None:
        return None, None
    if criterion == "center":
        c = RF // 2
        hits = {j for j in range(n_out) if (j + c) in pocket}
    else:
        hits = {j for j in range(n_out)
                if any((j + d) in pocket for d in range(RF))}
    return hits, len(pocket)


def selected_residues(alpha, n_out, topn=None):
    """Output residue indices carrying selection mass, per complex.

    alpha: (B, H, V, Q) or (B, V, Q). With topn, take the Q positions with
    the largest total absolute weight instead of the non-zero ones, which
    is how a dense pooled map is reduced to a matched support.
    """
    a = alpha if alpha.dim() == 4 else alpha.unsqueeze(1)
    # Sum |weight| over heads and atoms, leaving one score per residue.
    per_res = a.abs().sum(dim=(1, 2))[:, :n_out]          # (B, Q)
    out = []
    for b in range(per_res.shape[0]):
        row = per_res[b]
        if topn is None:
            out.append(set(torch.nonzero(row > 0).flatten().tolist()))
        else:
            k = min(int(topn[b]), int((row > 0).sum()), n_out)
            if k <= 0:
                out.append(set())
            else:
                out.append(set(row.topk(k).indices.tolist()))
    return out


def run_model(run_dir, split_csv, cfg, device, batches, dense_topn=None):
    """Yield (selected residue set, sequence, n_out) per complex."""
    model = IBDTA(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"),
                                     map_location=device))
    model.eval()
    df = pd.read_csv(split_csv)
    loader = make_loader(df, cfg, shuffle=False, verbose=False)
    max_len = cfg["PROTEIN"]["MAX_LEN"]
    # Sequences come from the dataset, not the dataframe. DTADataset drops
    # rows whose SMILES will not parse, so df.Protein and the loader can
    # differ in length and every sequence after the first drop would be
    # attributed to the wrong complex. Truncated the same way the
    # tokeniser truncates, so residue indices line up with the model's.
    seqs = [prot[:max_len].upper() for _, prot, _ in loader.dataset.rows]

    row = 0
    for i, batch in enumerate(loader):
        if batches and i >= batches:
            break
        nf, adj, nm, tok, pl = [b.to(device) if torch.is_tensor(b) else b
                                for b in batch[:5]]
        with torch.no_grad():
            o = model(nf, adj, nm, tok, pl)
        alpha = o["alpha"] if isinstance(o, dict) else o[2]
        B = alpha.shape[0]
        lens = pl.tolist() if torch.is_tensor(pl) else list(pl)
        # row is the index of this batch's first complex, so the matched
        # support slice and the sequence lookup both key off it. It is
        # advanced once per batch, after the yields: advancing it inside
        # the inner loop while also adding b double-counted the offset.
        topn = dense_topn[row:row + B] if dense_topn is not None else None
        sel = selected_residues(alpha.cpu(), alpha.shape[-1], topn)
        for b in range(B):
            n_out = max(int(lens[b]) - CROP, 1)
            yield sel[b], seqs[row + b], n_out
        row += B


def enrichment(run_dir, split_csv, cfg, device, batches, dense_topn=None,
               seed=0, criterion="overlap", strict=False):
    rng = random.Random(seed)
    rows, skipped = [], 0
    for sel, seq, n_out in run_model(run_dir, split_csv, cfg, device,
                                     batches, dense_topn):
        pock, n_pocket_res = pocket_output_indices(seq, n_out, criterion,
                                                   strict)
        if pock is None or not sel:
            skipped += 1
            continue
        sel = {j for j in sel if j < n_out}
        if not sel:
            continue
        chance = len(pock) / n_out
        hit = len(sel & pock)
        obs = hit / len(sel)
        # A matched random draw of the same size, as the floor the
        # enrichment has to clear on this very protein.
        rnd = set(rng.sample(range(n_out), min(len(sel), n_out)))
        rows.append({
            "support": len(sel),
            "n_out": n_out,
            "pocket_frac": chance,
            "in_pocket": obs,
            "enrichment": obs / chance if chance > 0 else float("nan"),
            "random_in_pocket": len(rnd & pock) / len(rnd),
            "p": hypergeom_sf(hit, n_out, len(pock), len(sel)),
        })
    out = pd.DataFrame(rows)
    # Carried so the summary can state how much of the split was scorable;
    # an enrichment without that number is not interpretable.
    out.attrs["skipped"] = skipped
    return out


def summarise(df, label):
    if df.empty:
        print(f"  {label}: no complexes scored")
        return None
    n = len(df)
    e = df.enrichment
    sk = df.attrs.get("skipped", 0)
    print(f"\n  {label}  ({n} complexes scored, {sk} skipped as "
          f"unresolved = {n / (n + sk):.1%} coverage)" if n + sk
          else f"\n  {label}")
    print(f"    mean support            {df.support.mean():8.1f} windows")
    print(f"    pocket fraction (chance){df.pocket_frac.mean():8.3f}")
    print(f"    selected in pocket      {df.in_pocket.mean():8.3f}")
    print(f"    random in pocket        {df.random_in_pocket.mean():8.3f}")
    ceiling = 1.0 / df.pocket_frac.mean() if df.pocket_frac.mean() else 0
    print(f"    enrichment              {e.mean():8.2f}x "
          f"(median {e.median():.2f}, sd {e.std():.2f})")
    print(f"    ceiling at this chance  {ceiling:8.2f}x "
          f"<- the most any method could score")
    print(f"    complexes enriched >1x  {(e > 1).mean():8.1%}")
    print(f"    complexes p<0.05        {(df.p < 0.05).mean():8.1%}")
    return {"n": n, "enrichment": float(e.mean()),
            "in_pocket": float(df.in_pocket.mean()),
            "random": float(df.random_in_pocket.mean()),
            "frac_enriched": float((e > 1).mean()),
            "frac_sig": float((df.p < 0.05).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="additive run directory (best.pt + "
                                  "config_used.json)")
    ap.add_argument("--control", help="pooled run directory, scored at the "
                                      "additive model's support")
    ap.add_argument("--audit", action="store_true",
                    help="motif detection rate only; no model needed")
    ap.add_argument("--data", default="davis")
    ap.add_argument("--root", default="./datasets")
    ap.add_argument("--batches", type=int, default=12,
                    help="0 for the whole split")
    ap.add_argument("--strict", action="store_true",
                    help="require all five motifs rather than HRD+DFG. "
                         "Lower coverage, larger pockets; report both if "
                         "you report either.")
    ap.add_argument("--criterion", default="both",
                    choices=["overlap", "center", "both"],
                    help="how a 16-residue window counts as in-pocket")
    ap.add_argument("--out", default=None, help="write per-complex CSV here")
    args = ap.parse_args()

    if args.audit:
        full = os.path.join(args.root, args.data, "full.csv")
        if not os.path.exists(full):
            raise SystemExit(f"missing {full}; run download_data.py first")
        seqs = pd.read_csv(full).Protein.drop_duplicates().tolist()
        print(f"\n  motif audit, {args.data}  [anchors: HRD+DFG required]")
        KM.audit(seqs)
        print(f"\n  motif audit, {args.data}  [strict: all five required]")
        KM.audit(seqs, strict=True)
        return

    if not args.run:
        raise SystemExit("--run or --audit required")
    cfg = json.load(open(os.path.join(args.run, "config_used.json")))
    meta = json.load(open(os.path.join(args.run, "test_metrics.json")))
    data, split = meta.get("data", args.data), meta["split"]
    folder = os.path.join(cfg.get("DATA_ROOT", args.root), data, split)
    csv = os.path.join(folder, "test.csv")
    if not os.path.exists(csv):
        csv = os.path.join(os.path.dirname(folder.rstrip("/\\")), "test.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  {data}/{split}   resolution {RF} residues per window")

    crits = (["overlap", "center"] if args.criterion == "both"
             else [args.criterion])
    ccfg = None
    if args.control:
        ccfg = json.load(open(os.path.join(args.control,
                                           "config_used.json")))
    for crit in crits:
        print(f"\n{'=' * 64}\n  criterion: {crit}\n{'=' * 64}")
        add = enrichment(args.run, csv, cfg, device, args.batches,
                         criterion=crit, strict=args.strict)
        s_add = summarise(add, "additive (ours)")
        if args.out:
            path = args.out.replace(".csv", f"_{crit}.csv")
            add.to_csv(path, index=False)
            print(f"\n    per-complex rows -> {path}")
        if ccfg is None:
            continue
        # Matched support, complex by complex, from the additive run.
        topn = add.support.tolist() if not add.empty else []
        pol = enrichment(args.control, csv, ccfg, device, args.batches,
                         dense_topn=topn, criterion=crit,
                         strict=args.strict)
        s_pol = summarise(pol, "pooled, top-N at matched support")
        if s_add and s_pol:
            print(f"\n  additive {s_add['enrichment']:.2f}x against pooled "
                  f"{s_pol['enrichment']:.2f}x at the same support, "
                  f"ceiling {1/add.pocket_frac.mean():.2f}x")
            print("  Both measured against each protein's own pocket "
                  "fraction, so the comparison is not confounded by "
                  "pocket size.")

    print("\n  Two caveats that belong with any number above.")
    print("  A motif-derived pocket is a biological prior, not a "
          "crystallographic measurement, so the claim is that the selected")
    print("  contacts concentrate on the catalytic machinery, not that "
          "they are the co-crystal contacts. And coverage is below 100%, "
          "so")
    print("  an absolute enrichment describes the resolvable subset while "
          "the additive-against-pooled comparison, run on identical")
    print("  sequences, does not depend on which subset that is. Lead with "
          "the comparison.")


if __name__ == "__main__":
    main()
