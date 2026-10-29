# IB-DTA — Interaction Bottlenecks for Drug–Target Prediction

Making the interaction map the **only** path to the prediction, so that an
explanation is the mechanism rather than a description of it.

---

## Base paper

This work extends **DrugBAN**:

> **Bai, P., Miljković, F., John, B. & Lu, H.** (2023).
> *Interpretable bilinear attention network with domain adaptation improves
> drug–target prediction.*
> **Nature Machine Intelligence** 5, 126–136.
> doi:[10.1038/s42256-022-00605-1](https://doi.org/10.1038/s42256-022-00605-1)
> · [paper](https://www.nature.com/articles/s42256-022-00605-1)
> · [preprint](https://arxiv.org/abs/2208.02194)
> · [code](https://github.com/peizhenbai/DrugBAN)

DrugBAN is the right starting point for three reasons: it is the canonical
model for explicit drug–target interaction modelling, its code is public
and runnable, and it is still the comparator recent work benchmarks against
(for example [Top-DTI](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i133/8199357),
*Bioinformatics* 2025).

```bibtex
@article{bai2023drugban,
  title   = {Interpretable bilinear attention network with domain adaptation
             improves drug-target prediction},
  author  = {Bai, Peizhen and Miljkovi{\'c}, Filip and John, Bino and Lu, Haiping},
  journal = {Nature Machine Intelligence},
  volume  = {5}, pages = {126--136}, year = {2023},
  doi     = {10.1038/s42256-022-00605-1}
}
```

---

## The idea in one diff

DrugBAN builds an explicit atom × residue interaction map, then collapses it
(`ban.py`, `attention_pooling`):

```python
fusion_logits = einsum('bvk,bvq,bqk->bk', v, att_map, q)   # map -> joint vector
```

After that contraction, no individual contact is on the critical path to the
output. The attention map *correlates* with the prediction; it does not
*produce* it. Every interpretable DTA model in this lineage shares that shape
and validates its explanations by visual overlap with known binding sites —
never by a causal test.

IB-DTA replaces the contraction with an additive decode over a sparse support:

```
ŷ  =  b  +  Σ(i,j)∈S  α_ij · g_θ(h_i, h_j)
```

No joint vector. Deleting a contact changes the output by **exactly** that
contact's contribution, so attributed importance and causal effect are the
same number by construction — verified numerically to 1.5 × 10⁻⁸.

Each selected pair carries a **signed contribution in pKd units**, so the
model states not just *where* it looks but *how much that contact is worth*.

---

## Quick start

```bash
pip install -r requirements.txt

python test_ib.py                                  # 8 property tests, ~30 s
python download_data.py --datasets davis           # 30,056 pairs, pKd
python splits.py --csv datasets/davis/full.csv --out datasets/davis
python train_ib.py --cfg configs/IBDTA.yaml --data davis --split cold_target
python run_experiments.py --quick                  # figures/ + results/
```

To reproduce the theory table without training anything, on CPU in about
half an hour:

```bash
python verify_theory.py            # propositions 1-9
python verify_generalization.py    # capacity vs budget and sample size
python verify_identifiability.py   # adversarial map under a pooled decoder
```

For numbers comparable with published Davis and KIBA tables, build the
canonical folds first and train against them:

```bash
python benchmark_split.py --datasets davis kiba
python train_ib.py --cfg configs/IBDTA.yaml --data davis --split benchmark/fold0
```

No `dgl` required — the default encoders are plain PyTorch. That removes the
dependency most likely to break a reproduction, and makes the Colab notebook
install in seconds.

**Colab:** `notebooks/IB_DTA_Colab.ipynb` — upload the zip, run top to bottom,
~10–15 min on a free CPU instance.

---

## What's here

| File | Purpose |
|---|---|
| `ib_layer.py` | `InteractionBottleneck`: α-entmax (vendored), sparse selection, additive pair decoder, adaptive budget, chunked decoding |
| `encoders.py` | dgl-free GCN + CNN encoders with **exact** padding masks; RDKit featurisation; dataset and collate |
| `models_ib.py` | `IBDTA` model, losses (MSE + ranking + sparsity), regression and classification metrics |
| `faithfulness.py` | Deletion-curve audit, AFS/polarity, additive-exactness check, adapters for baselines |
| `patch_ban.py` | Reversible monkey-patch so DrugBAN itself can be audited (no-op on the default path) |
| `splits.py` | random / cold-drug / cold-target / scaffold splits, leakage audit, **duplicate audit** |
| `download_data.py` | Davis, KIBA, BindingDB, BioSNAP, Human into one schema |
| `train_ib.py` | Training entry point |
| `run_experiments.py` | E0–E6 suite → `figures/`, `results/` |
| `benchmark_split.py` | Canonical DeepDTA fold protocol: 6 parts, 1 held out, 5-fold CV over the rest. Use this for numbers comparable with published tables |
| `design.py` | Matched molecular pairs via Bemis–Murcko scaffolds + MCS, atom-level attribution, counterfactual swap evaluation |
| `test_ib.py` | Property tests |
| `verify_numpy.py` | Independent NumPy cross-check of the core maths |
| `verify_theory.py` | Propositions 1–9: exactness, Shapley identity, marginals, composition |
| `verify_generalization.py` | Rademacher complexity against budget `k` and sample size `n` |
| `verify_identifiability.py` | Adversarial attention map under a pooled decoder; straight-through gradient alignment |
| `drugban_upstream/` | DrugBAN's original files, vendored unchanged, for the baseline audit. Needs `dgl` + `dgllife`; nothing else here does |
| `notebooks/` | Colab notebook |
| `figures/` | Generated figures (see below) |

---

## Verified

`test_ib.py` — **8/8 passing**, PyTorch 2.14.1:

| Property | Result |
|---|---|
| α-entmax yields exact zeros | 97.8% zeros vs 0.0% for softmax |
| **Additive deletion is exact** | max residual `1.49e-08` |
| Pooled decoder violates the same bound | residual `5.23` pKd — the bound does not hold |
| Deletion audit discriminates | AFS `+0.68` true vs `−0.04` random |
| Padding never selected | 0.0 mass on padded atoms/residues |
| Top-k budget respected | support = k exactly, every sample |
| Gradients reach `v_proj` | all three selection modes |
| NumPy ↔ PyTorch agreement | `< 6e-16` |
| Chunked vs dense decoding | bit-identical, 16× less memory |

---

## Findings from building this

**1. Two bugs that pass the obvious tests.**
α-entmax's bisection was inverted (`s(τ)` decreases in τ, so `s < 1` must
lower the upper bound). It still converged — to the wrong root — and returned
a **single non-zero entry regardless of input scale**. The model would have
trained, the loss would have fallen, sparsity would have looked spectacular,
and the bottleneck would have been an argmax keeping one contact. The
additive-exactness test passed throughout, because exactness is a property of
the *decoder*, not the *selection*.

Separately, support depended on embedding width (logit std ≈ 15 at
`H_DIM=16`, ≈ 133 at `H_DIM=128`), which would silently confound every budget
sweep. Fixed by standardising logits per (sample, head) over unpadded pairs.

After the fix, temperature maps predictably onto sparsity:

| temp | support |
|---|---|
| 0.5 | 1% |
| 0.7 | 3% |
| 1.0 | 15–17% |
| 1.5 | 66% |
| 2.0 | 83% |

**Watch `val_support` during training.** If it collapses toward 1, the
explanations are empty no matter what the loss says.

**2. Davis contains contradictory labels.**
Davis lists 442 protein *entries* but only 379 unique *sequences*. Isoforms
and mutants with identical sequences collapse onto the same key once you
index by sequence text — which is what a sequence-based model actually sees:

| | Davis | KIBA |
|---|---|---|
| duplicate (drug, sequence) groups | 1,224 | 473 |
| of those, with **disagreeing** labels | 546 | 288 |
| worst disagreement | 4.75 pKd | 2.78 |
| rows affected | 9.6% | 0.6% |
| implied MSE floor | **0.032** | 0.001 |

Published Davis MSE is typically 0.20–0.25, so roughly 13–16% of the
remaining error is irreducible for any sequence-based model. `splits.py`
reports this (`audit_duplicates`) and can collapse it (`deduplicate`).

**3. Random splits on Davis leak everything.**
The matrix is dense and there are only 68 distinct drugs, so a random split
puts **100% of test drugs and 100% of test targets** in training, plus 16.9%
outright duplicate pairs. Cold and scaffold regimes are the only meaningful
ones here.

**4. On Davis, the scaffold split is redundant.**
All 68 Davis compounds have **distinct** Bemis–Murcko scaffolds (no
scaffold is shared by two drugs), so a scaffold split reduces to a
cold-drug split — the two produce identical metrics, which is how the
redundancy was noticed. Use KIBA for a meaningful scaffold evaluation:
2,000 sampled compounds map to 1,160 scaffolds there. Reporting both
regimes on Davis as if they were independent evidence would be
double-counting.

**5. Chunking bounds inference memory, not training memory.**
`_pair_contributions` splits the pair decode over the residue axis, which
caps the live working set at `B x H x V x chunk x 2D`. Under autograd that
cap does not hold: every chunk's activations stay alive until backward, so
they add back up to the unchunked total. Measured at V=290, Q=1200, D=128,
B=4: **0.77 GB** with `no_grad` against **4.14 GB** with gradients. The
analytic figure describes inference and is the wrong number to size a GPU
against.

`IB.CHECKPOINT_CHUNKS` (default `true`) recomputes each chunk during
backward instead of storing it. That restores the bound during training for
one extra forward pass over the pair MLP:

| batch | checkpointing off | on |
|---|---|---|
| 4 | 4.14 GB | **1.05 GB** |
| 32 (default) | out of memory at 7 GB | **3.24 GB** |

Loss and summed parameter gradients are unchanged to ten and eight
significant figures, so this is exact, not an approximation. Without it the
default `BATCH_SIZE: 32` needs roughly 30 GB and will only fit on an A100
40GB or larger; with it the same config trains on a 12 GB card.

**6. The deletion test alone does not separate the two architectures.**
This one revised the research framing, and it only appeared once the
deletion fractions were log-spaced rather than linear.

| | IB-DTA (additive) | Pooled (DrugBAN-style) |
|---|---|---|
| AFS | **+0.758** | **+0.759** |
| polarity | +1.000 | +1.000 |
| additive bound | exact (`9.5e-07`) | violated (`4.79` pKd) |

Both models' attention magnitudes rank contacts by importance perfectly
well — the pooled model is *not* producing arbitrary explanations. So a
paper framed purely around "do deletion curves beat chance?" would find
no difference and have nothing to report.

What separates them is **calibration, not ranking**: the pooled model
cannot tell you *how much* a contact is worth, only which contacts matter
more than others. Its attributed magnitude is off by ~4.8 pKd, a 7-order-
of-magnitude gap in relative residual.

The implication for the paper: lead with **exactness**, and treat AFS as a
necessary-but-not-sufficient screen. Had the fractions stayed linear, the
step-function curves would have hidden this entirely.

---

## Sample results (demo scale — read the caveat)

From `run_experiments.py` with **200 training rows and 4 epochs on CPU**.
Concordance index sits at 0.47–0.55 across every variant, i.e. **near
chance**: at this scale nothing is really learning, so none of these
accuracy numbers mean anything. They are here to show the pipeline runs
end to end and produces every artefact.

| variant | MSE | CI | AFS | support |
|---|---|---|---|---|
| entmax + adaptive | 1.258 | 0.547 | +0.746 | 45.6 |
| entmax, fixed | 1.168 | 0.503 | +0.736 | 51.7 |
| top-k (k=16) | 1.063 | 0.478 | +0.757 | 16.0 |
| dense softmax | 1.061 | 0.472 | +0.739 | **12,480.7** |

The one shape worth noting — and worth *testing properly* at full scale —
is the last column against the first: a **780× smaller support** (16 pairs
vs 12,480) costs essentially nothing in MSE here. That is the frontier
question the research plan is built around. At this subset size it is a
hint, not evidence.

The budget sweep shows the same shape and locates the knee:

| budget k | MSE | AFS | support | deletion residual |
|---|---|---|---|---|
| 2 | 1.341 | +0.784 | 2.0 | `0.0e+00` |
| 4 | 1.096 | +0.762 | 4.0 | `0.0e+00` |
| **8** | **1.053** | +0.731 | 8.0 | `7.2e-07` |
| 16 | 1.063 | +0.757 | 16.0 | `8.3e-07` |
| *dense softmax* | *1.061* | *+0.739* | *12,480.7* | *violates the bound* |

MSE falls to k=8 and then flattens at the dense model's level. **Eight
contacts match twelve thousand** — while the dense model cannot state what
any one of them is worth. Exactness holds at every budget.

Caveat worth repeating: at CI ≈ 0.5 none of these models are learning, so
treat this as a demonstration that the sweep runs and produces the right
kind of curve. The sweep was cut at k=16 because the remaining budgets
exceeded the CPU budget here; `run_experiments.py` sweeps
`[2, 4, 8, 16, 32, 64]` by default on a GPU.

## Figures

Generated by `run_experiments.py`. All demo-scale unless noted — they show
the machinery works, not publishable numbers.

| Figure | Shows |
|---|---|
| `e0_leakage.png` | How much of each test set the model has already seen |
| `e1_splits.png` | Accuracy across the four split regimes |
| `e2_frontier.png` | Faithfulness–accuracy frontier over the budget sweep |
| `e3_deletion_curves.png` | Deletion curves: additive vs pooled |
| `e3_exactness.png` | Whether |Δprediction| equals attributed importance |
| `e4_support_dynamics.png` | Support size during training (collapse watch) |
| `e5_temperature.png` | Temperature → sparsity calibration (analytic, exact) |
| `e6_ablations.png` | Selection mode and adaptive budget |

---

## Auditing other models

The audit is model-agnostic and is the contribution that stands on its own —
it needs no new architecture, only the willingness to test published claims.

```python
import patch_ban; patch_ban.apply()          # makes DrugBAN deletable
import faithfulness as FA
from faithfulness import DrugBANAdapter

a = DrugBANAdapter(drugban_model, bg_d, v_p)
print(FA.format_report("DrugBAN", FA.audit(a.predict_fn(), a.importance())))
```

If published interpretable DTA models score AFS near zero, that is a result
whether or not IB-DTA wins on accuracy.

---

## Known limits


1. **Memory.** `IB.CHUNK_Q` bounds the pair decoder; lower it if you OOM.
   At V=290, Q=1000, D=128, B=64 the dense form needs ~19 GB, `CHUNK_Q=64`
   needs ~1.2 GB.
2. **Baselines must be rerun**, not quoted — published numbers use different
   splits and preprocessing.
3. **Heads.** More heads means a larger effective support and a weaker
   bottleneck. Keep `HEADS` at 1–2 unless ablating it.

---

## License

MIT — see `LICENSE`, which carries the DrugBAN attribution and dataset
provenance notices.
