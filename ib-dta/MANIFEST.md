# Package contents

```
ib-dta/
├── README.md                  start here
├── LICENSE                    MIT + DrugBAN attribution + dataset provenance
├── requirements.txt           no dgl needed for the default backend
├── run_all.sh                 end-to-end pipeline
│
├── ib_layer.py                THE CONTRIBUTION: sparse additive bottleneck
├── encoders.py                dgl-free GCN + CNN with exact padding masks
├── models_ib.py               IBDTA model, losses, metrics
├── faithfulness.py            deletion audit, AFS, additive-exactness check
├── patch_ban.py               makes upstream DrugBAN auditable (optional)
├── splits.py                  split regimes, leakage + duplicate audits
│
├── download_data.py           Davis, KIBA, BindingDB, BioSNAP, Human
├── train_ib.py                training entry point
├── run_experiments.py         E0-E6 suite -> figures/ + results/
│
├── test_ib.py                 8 property tests (run this first)
├── verify_numpy.py            independent NumPy cross-check
├── tools_temp_sweep.py        entmax sparsity diagnostic
│
├── configs/
│   ├── IBDTA.yaml             regression (pKd)
│   └── IBDTA_cls.yaml         binary DTI
│
├── notebooks/
│   └── IB_DTA_Colab.ipynb     upload + run, ~10-15 min on free CPU
│
├── figures/                   generated plots + README explaining each
└── results/                   generated CSV/JSON tables
```

## First five minutes

```bash
pip install -r requirements.txt
python test_ib.py                      # must be 8/8
python download_data.py --datasets davis
python splits.py --csv datasets/davis/full.csv --out datasets/davis
./run_all.sh                           # or run_experiments.py --quick
```
