"""
Minimal, reversible patch making DrugBAN auditable.

Upstream BANLayer.forward has no way to remove an interaction pair, so the
deletion test cannot be run against it as shipped. This monkey-patch adds an
optional `drop_pairs` argument that zeroes attention-map entries before
`attention_pooling`.

Why a patch and not a fork: the audit must run against DrugBAN *as published*.
Any change to how the model computes its prediction would invalidate the
comparison. This patch changes nothing on the default path -- with
drop_pairs=None the forward pass is byte-identical to upstream.

Usage
-----
    import patch_ban; patch_ban.apply()
    # DrugBAN(...) now accepts mode='eval', drop_pairs=<B,H,V,Q bool>
"""

import torch
import torch.nn as nn

# NOTE: `ban` and `models` are DrugBAN's upstream files and `models` imports
# dgllife at module level. Importing them here would make this module
# unimportable for anyone running the builtin (dgl-free) backend, which is
# the default. They are imported inside apply() instead, so the audit
# tooling stays usable without dgl installed.

_ORIGINAL = {}


def _patched_ban_forward(self, v, q, softmax=False, drop_pairs=None):
    v_num = v.size(1)
    q_num = q.size(1)
    if self.h_out <= self.c:
        v_ = self.v_net(v)
        q_ = self.q_net(q)
        att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
    else:
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
        d_ = torch.matmul(v_, q_)
        att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
        att_maps = att_maps.transpose(2, 3).transpose(1, 2)
    if softmax:
        p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
        att_maps = p.view(-1, self.h_out, v_num, q_num)

    # --- the only change ---------------------------------------------------
    if drop_pairs is not None:
        att_maps = att_maps * (~drop_pairs).to(att_maps.dtype)
    # -----------------------------------------------------------------------

    logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
    for i in range(1, self.h_out):
        logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
        logits += logits_i
    logits = self.bn(logits)
    return logits, att_maps


def _patched_drugban_forward(self, bg_d, v_p, mode="train", drop_pairs=None):
    v_d = self.drug_extractor(bg_d)
    v_p = self.protein_extractor(v_p)
    f, att = self.bcn(v_d, v_p, drop_pairs=drop_pairs)
    score = self.mlp_classifier(f)
    if mode == "train":
        return v_d, v_p, f, score
    elif mode == "eval":
        return v_d, v_p, score, att


def apply():
    """Install the patch. Idempotent.

    Requires DrugBAN's upstream ban.py / models.py on the path, which in
    turn require dgl + dgllife. Raises a clear error if they are missing
    rather than failing at import time.

    The upstream files live in drugban_upstream/, which is not a package
    and is therefore not importable by default. We put it on sys.path here
    so that `import ban` resolves without the caller having to.
    """
    if _ORIGINAL:
        return
    import os
    import sys
    vendored = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "drugban_upstream")
    if os.path.isdir(vendored) and vendored not in sys.path:
        sys.path.insert(0, vendored)
    try:
        import ban
        import models
    except ImportError as e:
        raise ImportError(
            f"patch_ban could not import DrugBAN's ban.py / models.py.\n"
            f"Looked in: {vendored}\n"
            f"Underlying error: {e}\n"
            "If the files are present, the cause is usually the missing "
            "dgl + dgllife dependency:\n"
            "    pip install dgl dgllife\n"
            "Only needed to audit DrugBAN itself; IB-DTA's own audit works "
            "without them.") from e
    _ORIGINAL["ban"] = ban.BANLayer.forward
    _ORIGINAL["drugban"] = models.DrugBAN.forward
    ban.BANLayer.forward = _patched_ban_forward
    models.DrugBAN.forward = _patched_drugban_forward


def revert():
    if not _ORIGINAL:
        return
    import ban
    import models
    ban.BANLayer.forward = _ORIGINAL.pop("ban")
    models.DrugBAN.forward = _ORIGINAL.pop("drugban")
