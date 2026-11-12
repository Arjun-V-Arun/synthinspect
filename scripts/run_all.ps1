# SynthInspect: the full experiment on Windows (RTX 4070 laptop).
# Run one block at a time; every stage resumes where it stopped, so it is safe to Ctrl+C and re-run.
# Start with ONE category (metal_nut) end to end, then add bottle and screw.

$ErrorActionPreference = "Stop"
$env:YOLO_OFFLINE = "1"
$cat = "metal_nut"

# 0. Setup (once)
# py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
# Install the CUDA build of PyTorch with the command pytorch.org/get-started gives for Windows + pip + CUDA
# (the plain PyPI wheel on Windows is CPU-only).
# pip install -e ".[gen,rfdetr,serve,dev]"
# Extract MVTec AD to data\mvtec_ad\ (see data\README.md)

# 1. Splits for every category
python -m synthinspect prepare

# 2. Synthetic defects (fast ones first)
python -m synthinspect generate --category $cat --method copypaste
python -m synthinspect generate --category $cat --method rgcb
python -m synthinspect generate --category $cat --method magic
python -m synthinspect generate --category $cat --method defectfill

# 3. Generator comparison: seed 0 for every condition first, then seeds 1 and 2
python -m synthinspect train --suite core --category $cat --seed 0
python -m synthinspect evaluate --suite core --category $cat
python -m synthinspect train --suite core --category $cat --seed 1 2
python -m synthinspect evaluate --suite core --category $cat
python -m synthinspect compare

# 4. Pretrained vs scratch vs RF-DETR (uses the best generator from step 3)
python -m synthinspect train --suite models --category $cat
python -m synthinspect evaluate --suite models --category $cat

# 5. Ship small, then explanations (the INT8 check needs the export)
python -m synthinspect export --category $cat
python -m synthinspect explain --category $cat

# 6. Report, demo assets, local API
python -m synthinspect report
python -m synthinspect demo --category $cat
# python -m synthinspect serve --model-dir runs\$cat\package
