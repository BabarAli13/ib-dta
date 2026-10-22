"""
IB-DTA model.

Base architecture: DrugBAN (Bai et al., Nature Machine Intelligence 2023)
    https://github.com/peizhenbai/DrugBAN

What is inherited unchanged
    - GCN over the molecular graph for atom embeddings
    - 3-layer CNN over the protein sequence for residue embeddings
    - the idea of an explicit atom x residue interaction map

What is replaced
    - DrugBAN contracts that map into a joint vector (bilinear pooling)
      before prediction. IB-DTA removes the contraction: affinity is a sum
      of signed contributions over a sparse selected set of pairs, so the
      interaction map is the only path to the output.

Holding the encoders fixed is deliberate. Any difference between the two
models is then attributable to the interaction module rather than to a
better representation.

Backends
    "builtin" (default) : pure-PyTorch encoders from encoders.py, no dgl
    "dgl"               : DrugBAN's original MolecularGCN / ProteinCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ib_layer import InteractionBottleneck


class IBDTA(nn.Module):
    def __init__(self, cfg=None, **kw):
        super().__init__()
        cfg = dict(cfg or {})
        cfg.update(kw)

        drug = cfg["DRUG"]
        prot = cfg["PROTEIN"]
        ib = cfg["IB"]
        self.task = cfg["TASK"]["NAME"]
        if self.task not in ("regression", "classification"):
            raise ValueError(
                f"TASK.NAME must be 'regression' or 'classification', got {self.task!r}")

        self.backend = cfg.get("ENCODER", {}).get("BACKEND", "builtin")

        if self.backend == "dgl":
            from models import MolecularGCN, ProteinCNN
            self.drug_encoder = MolecularGCN(
                in_feats=drug["NODE_IN_FEATS"],
                dim_embedding=drug["NODE_IN_EMBEDDING"],
                padding=drug["PADDING"],
                hidden_feats=drug["HIDDEN_LAYERS"])
            self.protein_encoder = ProteinCNN(
                prot["EMBEDDING_DIM"], prot["NUM_FILTERS"],
                prot["KERNEL_SIZE"], prot["PADDING"])
            drug_out = drug["HIDDEN_LAYERS"][-1]
            prot_out = prot["NUM_FILTERS"][-1]
        else:
            from encoders import GCNDrugEncoder, CNNProteinEncoder
            self.drug_encoder = GCNDrugEncoder(
                in_feats=drug["NODE_IN_FEATS"],
                dim_embedding=drug["NODE_IN_EMBEDDING"],
                hidden_feats=tuple(drug["HIDDEN_LAYERS"]),
                dropout=drug.get("DROPOUT", 0.0))
            self.protein_encoder = CNNProteinEncoder(
                embedding_dim=prot["EMBEDDING_DIM"],
                num_filters=tuple(prot["NUM_FILTERS"]),
                kernel_size=tuple(prot["KERNEL_SIZE"]),
                padding=prot["PADDING"])
            drug_out = self.drug_encoder.output_feats
            prot_out = self.protein_encoder.output_feats

        self.bottleneck = InteractionBottleneck(
            v_dim=drug_out, q_dim=prot_out,
            h_dim=ib["H_DIM"], heads=ib["HEADS"],
            selection=ib["SELECTION"], budget_k=ib["BUDGET_K"],
            adaptive_budget=ib["ADAPTIVE_BUDGET"],
            out_dim=1, dropout=ib["DROPOUT"],
            chunk_q=ib.get("CHUNK_Q", 0))

    # ---- forward ---------------------------------------------------------
    def forward(self, node_feats, adj, node_mask, tokens, lengths=None,
                drop_pairs=None):
        """Builtin backend signature. For the dgl backend pass
        (bg_d, None, None, v_p) and masks are derived internally."""
        if self.backend == "dgl":
            v_d = self.drug_encoder(node_feats)          # node_feats is bg_d
            v_p = self.protein_encoder(tokens)
            v_mask = torch.ones(v_d.shape[:2], dtype=torch.bool, device=v_d.device)
            q_mask = torch.ones(v_p.shape[:2], dtype=torch.bool, device=v_p.device)
        else:
            v_d = self.drug_encoder(node_feats, adj, node_mask)
            v_p, q_mask = self.protein_encoder(tokens, lengths)
            v_mask = node_mask

        y, contrib, alpha = self.bottleneck(
            v_d, v_p, v_mask=v_mask, q_mask=q_mask, drop_pairs=drop_pairs)
        return y, contrib, alpha

    def sparsity_penalty(self, alpha):
        return self.bottleneck.sparsity_penalty(alpha)

    @staticmethod
    def support_size(alpha):
        return InteractionBottleneck.support_size(alpha)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def regression_loss(y_pred, y_true, alpha=None, model=None,
                    lambda_rank=0.1, lambda_sparse=1e-3):
    """MSE + pairwise ranking + concave sparsity penalty.

    The ranking term is kept from DTA practice because virtual screening
    cares about order, not only magnitude.
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1).float()
    mse = F.mse_loss(y_pred, y_true)

    rank = y_pred.new_zeros(())
    if lambda_rank > 0 and y_pred.numel() > 1:
        dp = y_pred[:, None] - y_pred[None, :]
        dt = y_true[:, None] - y_true[None, :]
        mask = dt > 0
        if mask.any():
            rank = F.softplus(-dp[mask]).mean()

    sparse = y_pred.new_zeros(())
    if lambda_sparse > 0 and alpha is not None and model is not None:
        sparse = model.sparsity_penalty(alpha)

    total = mse + lambda_rank * rank + lambda_sparse * sparse
    return total, {"mse": float(mse.detach()), "rank": float(rank.detach()),
                   "sparse": float(sparse.detach())}


def classification_loss(y_logit, labels, alpha=None, model=None,
                        lambda_sparse=1e-3):
    y_logit = y_logit.view(-1)
    labels = labels.view(-1).float()
    bce = F.binary_cross_entropy_with_logits(y_logit, labels)
    sparse = y_logit.new_zeros(())
    if lambda_sparse > 0 and alpha is not None and model is not None:
        sparse = model.sparsity_penalty(alpha)
    total = bce + lambda_sparse * sparse
    return (torch.sigmoid(y_logit), total,
            {"bce": float(bce.detach()), "sparse": float(sparse.detach())})


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def concordance_index(y_true, y_pred, chunk=4096):
    """CI: fraction of correctly ordered pairs among pairs with differing truth.

    Chunked so it does not allocate an n x n matrix on full test sets.
    """
    y_true = torch.as_tensor(y_true).view(-1).float()
    y_pred = torch.as_tensor(y_pred).view(-1).float()
    n = y_true.numel()
    if n < 2:
        return float("nan")
    total = 0.0
    conc = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dt = y_true[s:e, None] - y_true[None, :]
        dp = y_pred[s:e, None] - y_pred[None, :]
        m = dt > 0
        if m.any():
            total += float(m.sum())
            conc += float((dp[m] > 0).sum()) + 0.5 * float((dp[m] == 0).sum())
    return conc / total if total else float("nan")


def regression_metrics(y_true, y_pred):
    import numpy as np
    yt = torch.as_tensor(y_true).view(-1).float().cpu().numpy()
    yp = torch.as_tensor(y_pred).view(-1).float().cpu().numpy()
    mse = float(((yt - yp) ** 2).mean())
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    out = {
        "mse": mse,
        "rmse": mse ** 0.5,
        "ci": concordance_index(yt, yp),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }
    if len(yt) > 1 and yt.std() > 0 and yp.std() > 0:
        out["pearson"] = float(np.corrcoef(yt, yp)[0, 1])
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(yt, yp).statistic)
    return out


def classification_metrics(y_true, y_prob):
    from sklearn.metrics import roc_auc_score, average_precision_score
    yt = torch.as_tensor(y_true).view(-1).cpu().numpy()
    yp = torch.as_tensor(y_prob).view(-1).cpu().numpy()
    out = {}
    if len(set(yt.tolist())) > 1:
        out["auroc"] = float(roc_auc_score(yt, yp))
        out["auprc"] = float(average_precision_score(yt, yp))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out
