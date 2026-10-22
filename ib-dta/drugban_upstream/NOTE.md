# Upstream DrugBAN files

Copied unmodified from https://github.com/peizhenbai/DrugBAN (MIT).

You need these only if you set `ENCODER.BACKEND = "dgl"` in a config, or
if you want to run the faithfulness audit against DrugBAN itself via
`patch_ban.py`. Both require `dgl` and `dgllife`.

The default `builtin` backend does not use anything in this folder.

To use them, copy them up one level so they sit beside `ib_layer.py`.
