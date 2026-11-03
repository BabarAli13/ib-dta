"""
Run the deletion audit on an already-trained checkpoint.

A hyperparameter search uses --no-audit, so its winners have weights but
no faithfulness measurement. Retraining one purely to audit it costs hours
for something that takes seconds: the checkpoint and the configuration
used to produce it are both already on disk.

    python audit_run.py --run tune/davis/benchmark/fold0_hdim256
    python audit_run.py --run <dir> --split val --batches 4

Reports AFS, polarity, the exactness residual and the mean support, and
says whether the attribution survives at whatever support the model
settled on. That last point is the reason this exists: a configuration
that wins a search by collapsing to a two-contact support has to be shown
to still explain its own predictions before it can be adopted.
"""
import argparse
import json
import os

import numpy as np
import torch

import faithfulness as FA
from models_ib import IBDTA
from train_ib import make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="a run directory containing best.pt and "
                         "config_used.json")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batches", type=int, default=12,
                    help="batches to audit. Support varies enough between "
                         "complexes that a few batches misestimate it: 3 "
                         "batches gave 18.4 contacts where the full test "
                         "set gave 27.3. AFS and exactness are stable much "
                         "sooner than support is.")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run, "config_used.json")
    ckpt = os.path.join(args.run, "best.pt")
    for p in (cfg_path, ckpt):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    cfg = json.load(open(cfg_path))
    meta_path = os.path.join(args.run, "test_metrics.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IBDTA(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"  {args.run}")
    print(f"  H_DIM={cfg['IB']['H_DIM']}  selection={cfg['IB']['SELECTION']}  "
          f"adaptive={cfg['IB']['ADAPTIVE_BUDGET']}  "
          f"MAX_LEN={cfg['PROTEIN']['MAX_LEN']}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    if meta.get("best_epoch"):
        print(f"  checkpoint from epoch {meta['best_epoch']}")

    data = meta.get("data", "davis")
    split = meta.get("split")
    if not split:
        raise SystemExit("test_metrics.json has no split; cannot locate data")
    folder = os.path.join(cfg.get("DATA_ROOT", "./datasets"), data, split)
    # Same lookup train_ib.py uses: a benchmark fold directory holds only
    # train and val, because all five folds share one independent test set
    # one level up.
    import pandas as pd
    csv = os.path.join(folder, f"{args.split}.csv")
    if not os.path.exists(csv):
        alt = os.path.join(os.path.dirname(folder.rstrip("/\\")),
                           f"{args.split}.csv")
        if args.split == "test" and os.path.exists(alt):
            csv = alt
        else:
            raise SystemExit(f"missing {csv}")
    df = pd.read_csv(csv)
    loader = make_loader(df, cfg, shuffle=False, verbose=False)

    afs, pol, res, sup, ok_all = [], [], [], [], True
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        nf, adj, nm, tok, pl = [b.to(device) if torch.is_tensor(b) else b
                                for b in batch[:5]]

        def predict(drop):
            with torch.no_grad():
                o = model(nf, adj, nm, tok, pl, drop_pairs=drop)
            return (o["y"] if isinstance(o, dict) else o[0]).squeeze(-1)

        with torch.no_grad():
            o = model(nf, adj, nm, tok, pl)
        contrib = o["contrib"] if isinstance(o, dict) else o[1]
        alpha = o["alpha"] if isinstance(o, dict) else o[2]
        if contrib.dim() == 3:
            contrib = contrib.unsqueeze(1)
        r = FA.audit(predict, contrib, fractions=FA.default_fractions(n=9),
                     seed=0)
        drop = torch.zeros_like(contrib, dtype=torch.bool)
        flat = contrib.reshape(contrib.shape[0], -1)
        idx = flat.abs().topk(min(5, flat.shape[1]), -1).indices
        drop.reshape(contrib.shape[0], -1).scatter_(1, idx, True)
        _, ok, det = FA.check_additive_exactness(predict, contrib, drop)
        afs.append(float(r["afs"]))
        pol.append(float(r["polarity"]))
        res.append(float(det["relative_residual"]))
        sup.append(float((alpha > 0).sum(dim=tuple(range(1, alpha.dim())))
                         .float().mean()))
        ok_all = ok_all and bool(ok)

    n_seen = len(afs) * cfg["SOLVER"]["BATCH_SIZE"]
    print(f"\n  audited {len(afs)} batch(es) of {args.split} "
          f"(~{n_seen} complexes)")
    print(f"  mean support      {np.mean(sup):8.1f} contacts "
          f"(sd {np.std(sup):.1f} across batches)")
    print(f"  AFS               {np.mean(afs):+8.4f}  "
          f"(min {min(afs):+.4f}, max {max(afs):+.4f})")
    print(f"  polarity          {np.mean(pol):8.4f}")
    print(f"  exactness holds   {ok_all}")
    print(f"  max rel residual  {max(res):8.2e}")

    # The question this script exists to answer.
    print()
    if np.mean(afs) < 0.3:
        print("  VERDICT: attribution has degraded. AFS below 0.3 means the "
              "selected contacts barely beat random deletion, so this "
              "configuration buys accuracy at the cost of the property the "
              "architecture exists for. Do not adopt it.")
    elif not ok_all:
        print("  VERDICT: exactness is violated, which should be impossible "
              "for the additive decoder. Investigate before trusting any "
              "number from this configuration.")
    else:
        print(f"  VERDICT: attribution is intact at a support of "
              f"{np.mean(sup):.1f} contacts. AFS {np.mean(afs):+.3f} and "
              f"exactness holds, so the accuracy gain is not bought by "
              f"giving up faithfulness.")


if __name__ == "__main__":
    main()
