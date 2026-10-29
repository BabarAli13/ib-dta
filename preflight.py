"""
Three-minute check that a session is set up correctly before committing
eleven hours to it.

    python preflight.py

Verifies, in the order things actually go wrong: the GPU is visible, the
repo is the version with the memory and early-stopping fixes, the modules
import, the data pipeline runs, and one real (tiny) training run completes
on the GPU including the deletion audit. Exits non-zero on the first
failure with a line saying what to do about it.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

FAIL = []


def check(name, ok, hint=""):
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    if not ok:
        if hint:
            print(f"          -> {hint}")
        FAIL.append(name)
    return ok


print("=" * 66)
print("  PREFLIGHT")
print("=" * 66)

# 1 ---------------------------------------------------------------- GPU
try:
    import torch
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"  torch {torch.__version__}   GPUs: {n}  {names}")
    check("CUDA available", torch.cuda.is_available(),
          "Settings -> Accelerator -> GPU. Without this every run is ~40x slower.")
    if n == 1:
        print("  note: one GPU visible; use the single-GPU command, not the"
              " two-process one.")
except ImportError as e:
    check("torch importable", False, str(e))

# 2 ------------------------------------------------------------- repo version
try:
    import yaml
    cfg = yaml.safe_load(open("configs/IBDTA.yaml"))
    check("IB.CHECKPOINT_CHUNKS present", "CHECKPOINT_CHUNKS" in cfg["IB"],
          "Old repo version. Without it batch 32 needs ~30 GB and will OOM.")
    pat = cfg["SOLVER"].get("PATIENCE")
    check("SOLVER.PATIENCE present", pat is not None,
          "Old repo version. Without it every run wastes ~1 h on dead epochs.")
    if pat:
        print(f"          patience = {pat}, max_epoch = {cfg['SOLVER']['MAX_EPOCH']}")
except Exception as e:
    check("configs readable", False, str(e))

check("run_davis.py present", os.path.exists("run_davis.py"),
      "Pull the latest repo.")
check("collect_results.py present", os.path.exists("collect_results.py"),
      "Pull the latest repo.")
if os.path.exists("run_davis.py"):
    src = open("run_davis.py").read()
    check("run_davis --only accepts several groups", 'nargs="*"' in src,
          "Old version; the two-GPU command will fail to parse.")

# 3 ------------------------------------------------------------- imports
for m in ("ib_layer", "encoders", "models_ib", "faithfulness", "splits"):
    try:
        __import__(m)
        check(f"import {m}", True)
    except Exception as e:
        check(f"import {m}", False, f"{type(e).__name__}: {e}")

# 4 ------------------------------------------------------------- data
have = os.path.isdir("datasets/davis") and os.path.exists(
    "datasets/davis/full.csv")
check("datasets/davis/full.csv", have,
      "python download_data.py --datasets davis   (needs Internet ON)")
for s in ("cold_target", "cold_drug", "random", "scaffold"):
    check(f"split {s}", os.path.isdir(f"datasets/davis/{s}"),
          "python splits.py --csv datasets/davis/full.csv --out datasets/davis")

# 5 ------------------------------------------- one real run, end to end
# Run this even if CUDA failed: it is the check that catches the most, and
# knowing the rest of the pipeline works is worth the extra minute on CPU.
blocking = [f for f in FAIL if f.startswith(("import", "split", "datasets"))]
if not blocking:
    print("\n  running one tiny training job "
          "(this is the part that catches everything else)...")
    tmp = tempfile.mkdtemp()
    cmd = [sys.executable, "train_ib.py", "--cfg", "configs/IBDTA.yaml",
           "--data", "davis", "--split", "cold_target", "--tag", "preflight",
           "--epochs", "1", "--subset", "64", "--eval-subset", "32",
           "--set", "RESULT.OUTPUT_DIR", tmp, "SOLVER.BATCH_SIZE", "8"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    ok = p.returncode == 0
    check("training job completes", ok, out.strip().splitlines()[-1] if out else "")
    if ok:
        if torch.cuda.is_available():
            check("ran on GPU", "device=cuda" in out,
                  "It fell back to CPU. Check the accelerator setting.")
        check("deletion audit ran", "AUDIT:" in out,
              "No AFS recorded; the frontier figure cannot be built.")
        mj = os.path.join(tmp, "davis", "cold_target_preflight",
                          "test_metrics.json")
        if check("test_metrics.json written", os.path.exists(mj)):
            d = json.load(open(mj))
            check("metrics contain afs", "afs" in d,
                  d.get("audit_error", "audit did not record AFS"))
            check("exactness holds", d.get("exact_ok") is True,
                  "The additive decode is not exact; stop and investigate.")
    shutil.rmtree(tmp, ignore_errors=True)

print("=" * 66)
if FAIL:
    print(f"  {len(FAIL)} CHECK(S) FAILED -- fix before Save & Run All:")
    for f in FAIL:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL CHECKS PASSED. Safe to Save & Run All.")
print("=" * 66)
