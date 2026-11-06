# IB-DTA — Interaction Bottlenecks for Drug–Target Prediction

Making the interaction map the **only** path to the prediction, so that an
explanation is the mechanism rather than a description of it.

---

## Status

Theory and dataset audits are measured and reproducible on CPU in about
half an hour. **Davis is complete:** 40 runs at the shipped configuration,
a hyperparameter sweep selected on validation only, and the five canonical
folds measured again at the one setting the sweep found. KIBA is stopped
part-way. Baselines have not started, and they are now the most important
thing outstanding, because they are what would turn the accuracy gap from
a bound into a measurement.

Every number in this file is measured at full scale. Nothing here is a
demonstration run.

### What this establishes

Three claims, in the order the evidence supports them.

1. **A pooled attention map cannot be read as an explanation.** Not
   "is often misleading": cannot. We construct, by direct optimisation, an
   attention map at total-variation distance 0.857 from a trained model's
   own map that reproduces its predictions to 1.3e-04. The displayed
   heatmap is one of many consistent with the same output.
2. **Replacing the contraction with an additive decode makes attribution
   exact.** Deleting any subset of contacts moves the prediction by the
   sum of their contributions, verified to 1.5e-08. Attributed importance
   and causal effect become the same number by construction rather than
   by correlation.
3. **The sparsity is free.** No budget from k=2 to k=64 is
   distinguishable from a dense model retaining all 21,012 contacts, which
   bounds the cost of a ten-thousandfold reduction in the interaction map
   below about 0.05 MSE.

What it costs is accuracy against tuned pooled models, measured and
reported below rather than in a footnote. The two are separable questions
and this file keeps them separate.

### Sparsity has no measurable accuracy cost

Cold-target, top-k against the dense control, three seeds per budget:

| k | support | MSE | vs dense |
|---|---|---|---|
| 2 | 2 | 0.5081 ± 0.0236 | +0.032 |
| 4 | 4 | 0.4924 ± 0.0277 | +0.017 |
| 8 | 8 | 0.4905 ± 0.0633 | +0.015 |
| 16 | 16 | 0.4891 ± 0.0299 | +0.013 |
| **32** | 32 | **0.4663 ± 0.0187** | −0.010 |
| 64 | 64 | 0.4725 ± 0.0329 | −0.003 |
| dense | **21,012** | 0.4759 ± 0.0341 | — |

All seven points now have three seeds.

**No budget differs significantly from dense, k=2 included: two contacts
against twenty-one thousand.** Dense ranks 3rd of 7, and two budgets beat
it numerically.

Read this as a bound rather than as a flat curve. The ordering of the
means does run monotonically with k, and a Spearman test on those six
means returns ρ = −0.943, p = 0.005. That test is the wrong one: it treats
each mean as a single observation and discards the sd ≈ 0.036 each one
carries. Regressing the 18 individual runs on log₂k gives a slope of
−0.0074 per doubling with p = 0.110 and a confidence interval spanning
zero. The total effect across a 32-fold range of k is 0.037 MSE against a
within-budget sd of 0.036, so three seeds cannot resolve it either way.

The claim the data supports: **any accuracy cost of reducing the support
from 21,012 contacts to 2 is below about 0.05 MSE.** Resolving whether the
true effect is zero or merely small would need roughly 15 seeds per
budget, about 90 runs, for a result that would still read "small".

Earlier revisions of this file described a knee at k=8. That came from
single-seed data: k=8 measured 0.5621 at one seed and 0.4416 at another, a
swing larger than the entire range across budgets. The knee was noise and
the description is withdrawn.

### Attribution

Across all 40 Davis runs: `exact_ok` true, worst
relative residual `1.8e-07`,
AFS between `+0.683` and `+0.792`.

**Polarity separates sparse from dense with no overlap.**

| selection | runs | polarity |
|---|---|---|
| entmax / top-k | 37 | exactly `1.0`, every run |
| dense softmax | 3 | all below `1.0` |

Polarity is `(AUC_most - AUC_least) / max(...)`, so it reaches exactly 1
only when deleting the *least* important contacts moves the prediction by
precisely zero. That requires those contacts to contribute exactly zero,
which is what α-entmax and top-k give and softmax structurally cannot.
The deletion audit detects the exact-zero property directly, rather than
inferring it from small weights.

Five top-k runs report a relative residual of exactly `0.0`. Those are the
k=2 and k=4 runs, where deleting the top five contacts deletes the whole
support.

Note that the dense control is softmax selection with the **additive**
decoder, so it satisfies exactness too. Exactness is a property of the
decoder, not of the selection. Demonstrating that a pooled decoder
violates it needs a genuinely pooled comparator, which means the
baselines, which have not been run.

### Split regimes

Davis, full scale, default configuration (adaptive entmax):

| Split | MSE | CI | r² | seeds |
|---|---|---|---|---|
| random | 0.3395 ± 0.0223 | 0.8464 ± 0.0102 | +0.57 | 3 |
| **cold-target** | **0.4846 ± 0.0429** | **0.7640 ± 0.0144** | +0.26 | 5 |
| cold-drug | 0.7563 ± 0.1652 | 0.6435 ± 0.0417 | −0.16 | 5 |
| scaffold | 1.1399 | 0.5973 | −0.75 | 1 |

### What it costs on accuracy

Canonical DeepDTA fold protocol, five folds. These are the only numbers
here directly comparable with published Davis tables, which is why they
are reported against the published models rather than against our own
retraining.

| Model | MSE | CI |
|---|---|---|
| NHGNN-DTA* | 0.196 | 0.914 |
| MGraphDTA | 0.207 | 0.900 |
| GraphDTA (GIN) | 0.229 | 0.893 |
| DeepDTA | 0.261 | 0.878 |
| **IB-DTA** (tuned) | **0.331 ± 0.035** | **0.850 ± 0.013** |
| IB-DTA (shipped config) | 0.362 ± 0.012 | 0.840 ± 0.005 |

<sub>* different split protocol, included because omitting the strongest
reported result would flatter ours.</sub>

That is 27% worse MSE than the oldest baseline and 60% worse than
MGraphDTA. We state it plainly because the alternative is for a reader to
discover it.

**The gap is not caused by the sparsity.** The dense control uses the same
architecture with no bottleneck at all, every one of the 21,012 contacts
retained, and scores 0.4759 on cold-target: *worse* than k=32 at 0.4663.
Removing the bottleneck recovers nothing. Whatever costs the accuracy sits
in the additive decoder or in the encoders, not in the selection.

So the two claims separate:

- **Sparsity is free.** Three seeds per budget, dense ranks 3rd of 7.
- **Interpretability by construction costs accuracy**, here +0.124 MSE
  against the strongest comparable pooled model.

What cannot yet be said is which of the two causes is responsible. The
additive constraint and the encoder generation are confounded, and
separating them needs pooled baselines trained under identical conditions.
Until that exists, the honest reading is that the gap is an upper bound on
the price of exact attribution, not a measurement of it.

### Tuning accounts for a fifth of the gap, and the sweep is incomplete

Single-factor variants on fold 0, selected on validation MSE only, epoch
ceiling 250, patience 25. The test set was never read during the sweep.

| configuration | val MSE | val CI | support | best epoch |
|---|---|---|---|---|
| shipped (h=128) | 0.3334 | 0.841 | 107.0 | 74 |
| λ_rank = 0.3 | 0.3459 | 0.832 | 32.8 | 80 |
| λ_rank = 0.8 | 0.3287 | 0.860 | 48.3 | 43 |
| h = 192 | 0.3118 | 0.856 | 43.4 | 80 |
| **h = 256** | **0.2899** | 0.854 | 27.1 | 116 |

Capacity is the only factor that moved the result beyond the seed-to-seed
sd of 0.043. Carried to all five folds it is worth 0.031 MSE, 0.362 to
0.331, at 556,163 parameters against roughly 1.5 M for DrugBAN. The wider
model is also **sparser**: 27 contacts against 107.

Two things this does not say. The improvement was still monotone at 256
and the 384 and 512 configurations did not run, so 256 is the best setting
*tested* and not an optimum. The learning rate, protein crop and sparsity
weight were never evaluated. The sweep was cut short by repeated
unexpected shutdowns on the machine running it, and the remaining trials
are queued rather than abandoned. Until they finish, the 0.124 bound is
loose by an unknown amount.

Patience 25 is itself suspect: the winning trial recorded its best at
epoch 116 and ran to 141, so it was stopped by the rule rather than by
convergence. `--patience` reruns the ladder at a looser value, tagging
results separately so the two protocols cannot be mixed.

### The tuned model keeps its attribution

Rerunning the deletion audit on all five tuned folds:

| | value |
|---|---|
| support | 40.6 ± 22.6 contacts of 21,012 |
| AFS | +0.746 ± 0.023 |
| exactness residual | 1.2e-07 worst, 2.1e-08 best |

The accuracy was not bought back from the property the architecture exists
for. Fold 1 is the outlier throughout: it stopped earliest at epoch 48,
settled on 80.1 contacts against 23.8 to 37.8 elsewhere, and returned the
worst MSE of the five at 0.389. The other four lie between 0.303 and
0.339, giving 0.316 ± 0.016 without it. We report the five-fold mean.

### Cold-drug is a negative result, not a bug

r² per seed: −0.577, −0.227, −0.069, +0.019, +0.038. At or below zero
means the model does no better than predicting the mean, and early
stopping fired as early as epoch 1. The model generalises to unseen
targets and not to unseen drugs. Davis has 68 compounds, so holding out 14
leaves 54 to learn chemistry from.

The scaffold split uses a byte-identical partition to cold-drug here, so
it is not independent evidence. The two runs landed 0.34 MSE apart, which
measures the instability of this regime rather than a difference between
regimes.

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
| `preflight.py` | Three-minute check that a session is set up correctly, before committing hours to it |
| `run_davis.py` | The full Davis queue, resumable across session-limited free GPUs |
| `run_tune.py` | Single-factor hyperparameter sweep, selected on validation only, `--patience` tags its own results |
| `run_final.py` | The five canonical folds at the tuned setting, with the audit on |
| `audit_run.py` | Deletion audit on an existing checkpoint, in seconds rather than retraining for hours |
| `keepawake.py` | Stops Windows sleeping underneath an unattended run |
| `collect_results.py` | Turns finished runs into summary tables and LaTeX fragments |
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

## Reproducing the full experiments

The Davis queue is 40 runs: three split regimes at five seeds (three
for random), one scaffold run, a seven-point budget sweep at three seeds,
and five canonical folds. Roughly 100 GPU-hours on a T4. `run_davis.py`
skips runs that already have a `test_metrics.json`, so re-running the same
command in a new session continues rather than restarts.

```bash
python preflight.py                       # 3 min; do this first
python download_data.py --datasets davis
python splits.py --csv datasets/davis/full.csv --out datasets/davis
python benchmark_split.py --datasets davis
python run_davis.py --hours 11 --out ./results
python collect_results.py --out ./results
python run_davis.py --list --out ./results     # what is left
```

Then the tuning pass and the final folds:

```bash
python run_tune.py                             # sweep, validation only
python run_tune.py --report                    # the table above
python run_final.py                            # five folds at the winner
python run_final.py --list                     # the paper row, formatted
```

`run_tune.py` never evaluates the test set (`--skip-test`), so a tuning
log cannot influence the choice even by being read. `run_final.py` turns
the test evaluation and the deletion audit back on.

On Windows, run long jobs through Task Scheduler as SYSTEM rather than
from a console. A console job dies with its login session, and the
machine will sleep underneath it unless something holds a power request;
`keepawake.py` asserts one, which `powercfg /requests` can confirm.

Split by group across two GPUs with `--only seeds` and
`--only frontier benchmark scaffold`. In a notebook the whole launch must
be one line: each `!` runs its own shell, so a `wait` on its own line has
no children and the session ends, killing both jobs.

Each run writes `test_metrics.json` (with AFS and the exactness residual),
`history.csv`, `config_used.json` and `best.pt`. Throughput is 176 s/epoch
on a T4 at full scale.

---

## Figures

Generated by `run_experiments.py` from whatever runs are present in
`results/`. The paper's figures are built separately by
`make_measured.py`, which reads only the full-scale summary and refuses to
invent anything.

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


1. **Memory.** `IB.CHUNK_Q` bounds the pair decoder at inference; during
   training you also need `IB.CHECKPOINT_CHUNKS: true`, which is the
   default. See finding 5 — the analytic figure is the wrong number to
   size a GPU against. If you still OOM, lower `CHUNK_Q` or the batch size.
2. **Baselines are the main gap.** No baseline implementation ships here,
   so the controlled-retraining comparison cannot be produced from this
   repo. That now matters more than it did: IB-DTA is 27-60% behind
   published models on the canonical protocol, tuning has been shown to
   account for about a fifth of that, and without pooled baselines trained
   under identical conditions the rest cannot be separated between the
   additive constraint and the encoder generation.
3. **Heads.** More heads means a larger effective support and a weaker
   bottleneck. Keep `HEADS` at 1–2 unless ablating it.
4. **Early stopping is a protocol choice, not a convenience.** On
   cold-target at the shipped width the best epoch was 17 of 50 and
   validation MSE rose monotonically from about epoch 23, so a tight
   patience costs nothing there. At width 256 the opposite holds: the best
   epoch was 116 and the run was stopped at 141 by patience rather than by
   convergence. A result measured at one patience is not comparable with
   one measured at another, so `run_tune.py` and `run_final.py` both put
   the value in the run's tag and write to separate directories.
5. **Seed counts vary and are stated.** The split regimes carry three to
   five seeds, the budget sweep three per point, and the scaffold row one.
   `collect_results.py` marks any cell backed by fewer than five.
6. **The sweep ran one seed per configuration.** That is enough to rank
   candidates coarsely and not enough to trust the ranking: the measured
   seed-to-seed sd of 0.043 is wider than most of the spread across the
   sweep. The capacity result survives because it was confirmed on all
   five folds afterwards, which is a paired comparison on the same data
   rather than one draw. `--confirm` reruns a named configuration across
   seeds for anything where that confirmation is not available.

---

## License

MIT — see `LICENSE`, which carries the DrugBAN attribution and dataset
provenance notices.
