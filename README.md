# SynthInspect

**Few-shot visual inspection with synthetic defects, explained and shipped small.**

A new part goes into production with hundreds of good samples but only five to eight photos of each defect. This project asks which way of making synthetic defects helps a detector most, measured strictly:

1. **The RGCB approach**: real defects pasted onto clean parts, then the whole image restyled with photorealistic HRNet style transfer, as in the generator I used at RGCB in 2024.
2. **DefectFill** (CVPR 2025).
3. **MAGIC** (CVPR 2026 Findings).

The resulting detector is then explained, quantised to INT8, served as an API, and run in the browser.

It mirrors how industrial inspection teams describe their own work: generative AI makes synthetic images for optical inspection, the model decides on its own and sends only doubtful parts to people, and large models are distilled into small ones that run on the device.

## Results

<!-- results:start -->
*Run the pipeline to fill this table (`python -m synthinspect report`).*
<!-- results:end -->

## What Is Compared

| Question | Conditions | Where |
|---|---|---|
| Which synthetic data helps most? | real only · copy-paste · RGCB (copy-paste + NST) · DefectFill · MAGIC | `suites.core`, 3 seeds, paired bootstrap tests |
| Pretrained or from scratch? Which detector? | YOLO26n COCO-pretrained · YOLO26n from scratch · RF-DETR Nano | `suites.models` |
| Where does the model look, and is that faithful? | ODAM vs D-RISE, scored against pixel-level defect masks | `explain` |
| Does INT8 change accuracy or explanations? | PyTorch FP32 · ONNX · OpenVINO FP16/INT8 · ONNX INT8 | `export`, `explain` |
| How much can be automated safely? | Pass / review / reject band, thresholds cross-fitted | every evaluation |

All generators paint into the same masks, drawn only from the training defects. This fixes a test-mask leak in DefectFill's published protocol and makes the comparison about how each method paints, not where.

The literature review and the reasons for each pick are in [docs/LITERATURE.md](docs/LITERATURE.md).

## Design Choices Worth Knowing

- **One backbone for both diffusion methods** (SD 1.5 inpainting). The official SD 2 repositories return 404 since November 2025.
- **MAGIC uses LoRA instead of full DreamBooth**, so it fits an 8 GB laptop GPU. This is a recorded deviation.
- **No checkpoint selection on any validation data.** Fixed epochs, last checkpoint.
- **INT8 keeps the detection head in FP32**, is calibrated on training images only, and must pass a release gate (at most 0.02 image AP lost).
- **The demo runs in the browser** (ONNX Runtime Web, WASM) from GitHub Pages: free, always on, and a real edge deployment. The API is a PyTorch-free container for Cloud Run. Details in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Run It

Windows PowerShell, an NVIDIA GPU with 8 GB, Python 3.12. The full sequence is in `scripts/run_all.ps1`.

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
# PyTorch with CUDA: use the pytorch.org/get-started command for Windows + pip
pip install -e ".[gen,rfdetr,serve,dev]"
# Extract MVTec AD into data\mvtec_ad\ (see data\README.md)

python -m synthinspect prepare
python -m synthinspect generate --category metal_nut --method rgcb
python -m synthinspect train --suite core --category metal_nut --seed 0
python -m synthinspect evaluate --suite core --category metal_nut
python -m synthinspect report
```

Every stage resumes where it stopped. All settings live in `configs/experiment.yaml` and `configs/generators.yaml`.

## Tests

`pytest -q` runs every stage on CPU in a few minutes, with no downloads. It uses a tiny MVTec-shaped fixture, a tiny random diffusion UNet and a few-epoch YOLO26. The tests cover:

- **The split:** leakage checks.
- **Mask placement:** placement rules hold.
- **Generators:** every DefectFill loss term reaches the LoRA weights, including through gradient checkpointing.
- **Metrics:** AP matches scikit-learn; box AP, AUROC and bootstrap intervals are checked on known cases.
- **Review band:** respects its targets.
- **Explanations:** recover a known answer.
- **Exports:** agree with PyTorch.
- **The API:** serves the packaged model.
- **The browser demo's JavaScript:** gives the same boxes and decisions as Python under the same WASM runtime.

## Repository Map

```
configs/      experiment and generator settings
docs/         LITERATURE.md · HOW_IT_WORKS.md · DEPLOYMENT.md
synthinspect/
  data/       split, foreground, mask bank, YOLO-format writer
  gen/        copypaste, rgcb (+nst), sd_inpaint (DefectFill and MAGIC), generation loop
  detect/     YOLO26 and RF-DETR training, one letterbox, ONNX/OpenVINO runtimes
  eval/       metrics with bootstrap CIs, review band with cross-fitting
  xai/        ODAM, D-RISE, explanation metrics, the explanation studies
  deploy/     export and benchmark
  serve/      FastAPI app, monitoring, demo packaging
web/          static browser demo (index.html, app.js, inspect.js)
tests/        CPU tests for every stage
```

## Limitations

- Three MVTec categories, one dataset. The defects are real but the parts are not Bosch parts.
- The re-implementations follow the papers' text. Where a paper leaves a value open, or its text and equations disagree, the choice is recorded in `configs/generators.yaml` and `docs/LITERATURE.md`.
- With about 60 defective test parts per category, confidence intervals are wide; read the paired tests, not single numbers.

## Licence

Code is AGPL-3.0 (it builds on Ultralytics YOLO26). MVTec AD and everything derived from it, including synthetic images, trained weights and demo assets, is CC BY-NC-SA 4.0: non-commercial only. See `LICENSE`.
