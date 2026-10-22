"""
Self-contained drug and protein encoders with no DGL dependency.

Why this exists
---------------
DrugBAN's encoders rely on `dgl` + `dgllife`. Those are the single most
fragile part of reproducing it: the wheels are pinned to specific
torch/CUDA combinations, they break on Colab regularly, and a mismatch
produces segfaults rather than clean errors.

Everything DrugBAN's `MolecularGCN` does can be written in ~40 lines of
plain PyTorch, because it is a vanilla GCN over a molecular graph. This
module does that, keeping the same 75-dimensional atom featurisation so
results remain comparable, and adds the thing DrugBAN never needed but
IB-DTA does: **exact padding masks**.

Masks matter here in a way they do not for a pooled model. DrugBAN sums
over everything, so padding contributes a roughly constant bias that the
network learns around. IB-DTA *selects* pairs, so an unmasked padding
position can win the selection and the resulting "explanation" points at
a residue that does not exist. A wrong mask does not degrade IB-DTA's
accuracy much; it silently destroys its interpretability, which is the
whole point of the model.

If `dgl`/`dgllife` are installed and you prefer the original path,
`models_ib.py` will use them when `ENCODER.BACKEND = "dgl"`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    _HAS_RDKIT = False


# --------------------------------------------------------------------------
# Featurisation
# --------------------------------------------------------------------------
ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
]

# DrugBAN uses dgllife's CanonicalAtomFeaturizer: 74 features + 1. We
# reproduce the same layout so a model trained with either backend sees
# inputs of the same width and meaning.
ATOM_FEAT_DIM = 75

CHARSET = {c: i + 1 for i, c in enumerate(
    "ABCDEFGHIKLMNOPQRSTUVWXYZ")}          # 0 is reserved for padding
MAX_PROTEIN_LEN = 1200
MAX_ATOMS = 290


def _onek(x, allowable, unknown_last=True):
    if x not in allowable:
        if unknown_last:
            x = allowable[-1]
        else:
            return [0] * len(allowable)
    return [int(x == a) for a in allowable]


def atom_features(atom):
    """75-dim canonical atom featurisation (matches dgllife's layout)."""
    return np.array(
        _onek(atom.GetSymbol(), ATOM_VOCAB + ['Unknown'])
        + _onek(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + _onek(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
        + _onek(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + _onek(int(atom.GetHybridization()), [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetIsAromatic()]
        + _onek(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + [atom.IsInRing()],
        dtype=np.float32,
    )


def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    """SMILES -> (node_features [N,75], adjacency [N,N], n_atoms).

    Returns None for unparseable SMILES so callers can drop the row rather
    than train on a silently-zeroed molecule.
    """
    if not _HAS_RDKIT:
        raise ImportError("rdkit is required: pip install rdkit")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    n = min(mol.GetNumAtoms(), max_atoms)
    feats = np.zeros((max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        if i >= n:
            break
        f = atom_features(atom)
        feats[i, :len(f)] = f[:ATOM_FEAT_DIM]

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < n and j < n:
            adj[i, j] = adj[j, i] = 1.0
    adj[np.arange(n), np.arange(n)] = 1.0      # self loops

    return feats, adj, n


def protein_to_ints(seq, max_len=MAX_PROTEIN_LEN):
    """Protein sequence -> (int tokens [L], true_length). 0 = padding."""
    x = np.zeros(max_len, dtype=np.int64)
    seq = seq[:max_len].upper()
    for i, ch in enumerate(seq):
        x[i] = CHARSET.get(ch, 0)
    return x, min(len(seq), max_len)


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------
class GCNDrugEncoder(nn.Module):
    """Plain GCN over a dense adjacency. Equivalent in spirit to DrugBAN's
    `MolecularGCN` but with no dgl dependency and explicit masking."""

    def __init__(self, in_feats=ATOM_FEAT_DIM, dim_embedding=128,
                 hidden_feats=(128, 128, 128), dropout=0.0):
        super().__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        dims = [dim_embedding] + list(hidden_feats)
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.dropout = nn.Dropout(dropout)
        self.output_feats = dims[-1]

    def forward(self, node_feats, adj, node_mask):
        """
        node_feats : (B, N, in_feats)
        adj        : (B, N, N)  with self loops
        node_mask  : (B, N) bool
        returns    : (B, N, output_feats), padded rows zeroed
        """
        m = node_mask.unsqueeze(-1).float()
        h = self.init_transform(node_feats) * m

        # symmetric normalisation D^-1/2 A D^-1/2
        deg = adj.sum(-1).clamp_min(1.0)
        dinv = deg.pow(-0.5)
        norm_adj = adj * dinv.unsqueeze(-1) * dinv.unsqueeze(1)

        for layer in self.layers:
            h = torch.bmm(norm_adj, h)
            h = F.relu(layer(h))
            h = self.dropout(h) * m
        return h


class CNNProteinEncoder(nn.Module):
    """Three Conv1d blocks, as in DrugBAN, with the output mask computed
    exactly rather than inferred.

    Each conv has no padding and kernel k, so a length-L input becomes
    L-(k-1). Tracking that arithmetic is the difference between masking
    correctly and guessing.
    """

    def __init__(self, embedding_dim=128, num_filters=(128, 128, 128),
                 kernel_size=(3, 6, 9), vocab=26, padding=True):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab, embedding_dim, padding_idx=0 if padding else None)
        ch = [embedding_dim] + list(num_filters)
        self.kernels = list(kernel_size)
        self.convs = nn.ModuleList(
            nn.Conv1d(ch[i], ch[i + 1], self.kernels[i]) for i in range(3))
        self.bns = nn.ModuleList(nn.BatchNorm1d(ch[i + 1]) for i in range(3))
        self.output_feats = ch[-1]

    def out_len(self, in_len):
        for k in self.kernels:
            in_len = in_len - (k - 1)
        return max(in_len, 0)

    def forward(self, tokens, lengths=None):
        """
        tokens  : (B, L) long
        lengths : (B,) true sequence lengths. If None, inferred from non-zero
                  tokens -- correct for our tokenizer, where 0 is padding only.
        returns : (B, L', C), mask (B, L') bool
        """
        if lengths is None:
            lengths = (tokens != 0).sum(dim=1)
        h = self.embedding(tokens.long()).transpose(2, 1)
        for conv, bn in zip(self.convs, self.bns):
            h = bn(F.relu(conv(h)))
        h = h.transpose(2, 1)                           # B, L', C

        crop = sum(k - 1 for k in self.kernels)
        # A sequence shorter than crop+1 leaves no valid positions. Rather
        # than emit an all-False mask (which makes the sample unpredictable
        # and its explanation empty), keep one position and say so once.
        out_lens = lengths - crop
        if bool((out_lens < 1).any()):
            if not getattr(self, "_warned_short", False):
                import warnings
                n_bad = int((out_lens < 1).sum())
                warnings.warn(
                    f"{n_bad} sequence(s) shorter than {crop + 1} residues: the "
                    f"CNN stack crops {crop}. Keeping 1 position so they remain "
                    f"predictable, but their explanations are meaningless. "
                    f"Filter short sequences upstream.", RuntimeWarning)
                self._warned_short = True
            out_lens = out_lens.clamp_min(1)
        out_lens = out_lens.clamp(min=0, max=h.shape[1])

        ar = torch.arange(h.shape[1], device=h.device)[None, :]
        mask = ar < out_lens[:, None]
        return h * mask.unsqueeze(-1).float(), mask


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class DTADataset(torch.utils.data.Dataset):
    """CSV-backed dataset. Expects columns SMILES, Protein, Y.

    Unparseable SMILES are dropped at construction with a count reported,
    rather than silently becoming zero graphs.
    """

    def __init__(self, df, max_atoms=MAX_ATOMS, max_protein=MAX_PROTEIN_LEN,
                 verbose=True):
        self.max_atoms = max_atoms
        self.max_protein = max_protein
        self.rows, dropped = [], 0
        for _, r in df.iterrows():
            g = smiles_to_graph(r["SMILES"], max_atoms)
            if g is None:
                dropped += 1
                continue
            self.rows.append((g, r["Protein"], float(r["Y"])))
        if verbose and dropped:
            print(f"  dropped {dropped}/{len(df)} rows with unparseable SMILES")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        (feats, adj, n), prot, y = self.rows[i]
        tok, plen = protein_to_ints(prot, self.max_protein)
        node_mask = np.zeros(self.max_atoms, dtype=bool)
        node_mask[:n] = True
        return (torch.from_numpy(feats), torch.from_numpy(adj),
                torch.from_numpy(node_mask), torch.from_numpy(tok),
                torch.tensor(plen), torch.tensor(y, dtype=torch.float32))


# Three convs with kernels (3,6,9) each crop (k-1): a sequence shorter than
# 1 + sum(k-1) = 17 residues produces ZERO valid output positions. Such a
# sample has no selectable pairs at all, so the model returns its bias and
# the "explanation" is empty -- silently, unless we guard for it. Real
# proteins are hundreds of residues so this only bites on toy data and
# truncated fragments, which is exactly where it goes unnoticed.
MIN_PROTEIN_LEN = 32


def collate(batch, min_protein=MIN_PROTEIN_LEN):
    """Trim to the largest molecule/protein in the batch -- the single
    cheapest speedup available, since padding to 290 atoms and 1200
    residues when the batch holds 40 and 300 wastes most of the compute.

    Never trims below `min_protein`, so the conv stack always has something
    left to work with.
    """
    feats, adj, nmask, tok, plen, y = zip(*batch)
    n_max = max(max(int(m.sum()) for m in nmask), 1)
    p_max = max(max(int(p) for p in plen), min_protein)
    return (
        torch.stack(feats)[:, :n_max],
        torch.stack(adj)[:, :n_max, :n_max],
        torch.stack(nmask)[:, :n_max],
        torch.stack(tok)[:, :p_max],
        torch.stack(plen),
        torch.stack(y),
    )
