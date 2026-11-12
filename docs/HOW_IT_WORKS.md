# How It Works

A guided read of the code, in the order the experiment runs. Read this before running anything; it is also your interview preparation.

## The Experiment in One Picture

```
MVTec AD category
  ├─ split (data/mvtec.py)          first 1/3 of each defect type = the few real shots; rest = test
  ├─ mask bank (data/maskbank.py)   where a synthetic defect may go; built ONLY from training defects
  ├─ generators (gen/)              same masks, four ways to paint a defect into them
  │    copypaste ─ rgcb (+NST) ─ defectfill (SD inpainting + LoRA) ─ magic (SD inpainting + LoRA, GPP, SAG)
  ├─ datasets (data/yolo_format.py) real defects + good parts (+ synthetic) in YOLO format
  ├─ detectors (detect/)            YOLO26n pretrained / from scratch, RF-DETR Nano
  ├─ evaluation (eval/)             image AP/AUROC, box AP, bootstrap CIs, paired tests, review band
  ├─ explanations (xai/)            ODAM + D-RISE, scored against the true defect masks
  ├─ export (deploy/)               ONNX, OpenVINO FP16/INT8, ONNX INT8; size, latency, accuracy
  └─ serving (serve/, web/)         FastAPI + Docker; static browser demo
```

## 1. The Split and Why It Is Strict

`make_split` sorts each defect type's files and takes the first `floor(n/3)` for training. That matches the protocol in AnomalyDiffusion and MAGIC, so it is defensible. `Split.check_disjoint()` raises if any image is on both sides, and a test checks it.

Three places where leakage usually sneaks in, and how each is closed:

| Leak | Where it happens elsewhere | Here |
|---|---|---|
| Generating on test masks | DefectFill's protocol (MAGIC and OSAGEN call it out) | The mask bank only reads training defects |
| Picking the best checkpoint on test data | Early stopping on a "val" set that is really test | Fixed epochs, last checkpoint, `val=False` |
| Choosing thresholds on the parts you then score | Most "automation rate" claims | Two-fold cross-fitting, repeated 20 times |

## 2. The Mask Bank

Every generator paints into the same kind of region, so the comparison is about how each one paints, not where. A training mask is flipped, rotated and scaled, then placed on a new good part, and kept only if at least 90% of it lands on the part. That 90% rule is MAGIC's own alignment score, with a simple colour-distance foreground (`data/foreground.py`) instead of U2-Net and keypoint matching.

Category knowledge lives in the config, not the code. For example, `metal_nut` has `flip: false`, because a mirrored nut is itself one of the defect types.

## 3. The Four Generators

**Copy-paste** (`gen/copypaste.py`) moves a real defect's pixels with exactly the transform used for its mask, then blends with Poisson seamless cloning. A test caught that a cloning margin under about 6 pixels makes OpenCV erase the defect's edge, and the defect fades away; the margin is now enforced.

**RGCB** (`gen/rgcb.py`, `gen/nst.py`) is copy-paste plus Gatys neural style transfer toward a real image of the same part, starting from the composite so structure (and the label) is kept. This is the idea from your RGCB internship moved to a new domain. **Check `configs/generators.yaml` → `rgcb` and set `scope` and `style_source` to match what you actually did there.**

**DefectFill** (`gen/sd_inpaint.py`, `gen/sd_generators.py`) fine-tunes SD inpainting with LoRA so the token `sks` means "this defect". Three losses:

- **Defect loss:** noise prediction inside the defect only.
- **Object loss:** the part with random boxes cut out, background down-weighted, so the model learns the defect in context.
- **Attention loss:** pushes the cross-attention map of `sks` onto the defect mask.

At generation it makes four candidates and keeps the one that differs most from the clean part (LPIPS).

**MAGIC** uses the same backbone, but trains on the whole image with noise added to the prompt embedding (training and generation), and generates with stronger guidance outside the defect than inside it early in sampling.

Two bugs worth knowing about, both caught by the tests on a tiny random UNet:

- **Gradient checkpointing with the attention recorder.** Checkpointing re-runs the forward pass during backward. The attention recorder was switched off before backward, so the re-run did different work and PyTorch refused. It now stays on until backward finishes.
- **Seeded generation was not repeatable.** The VAE encoder sampled its posterior with the global random state. Generation now uses the posterior mode.

## 4. Detectors and the "Pretrained vs Scratch" Question

`detect/yolo.py` trains YOLO26n from `yolo26n.pt` (COCO-pretrained) or from `yolo26n.yaml` (random weights). All runtimes share one letterbox (`detect/base.py`), so PyTorch, ONNX, OpenVINO and the browser can be compared box for box.

YOLO26's one-to-one head is NMS-free. Exporting with `nms=False` keeps it, so the ONNX output is final boxes `(1, 300, 6)`. Without that flag, Ultralytics exports the one-to-many head, which needs NMS. A test caught this.

RF-DETR Nano (`detect/rfdetr_wrap.py`) is Apache-licensed and built on DINOv2. Its classifier has one extra logit slot beyond the real classes. Comparing against `rfdetr`'s own `predict` showed that slot is the no-object score, so the decoder ignores it.

## 5. Metrics and the Review Band

- **Part score** = the most confident detection. Image AP and AUROC treat "defective" as positive.
- **Box AP** is reported at IoU 0.5 and 0.1, because a box around a thin scratch is mostly background.
- **Review band** (`eval/review.py`): two thresholds on the part score.
  - Below `t_pass`: auto-pass. This is set so no calibration defect slips through.
  - Above `t_reject`: auto-reject. This is set so at most 2% of calibration good parts are scrapped.
  - In between: a person looks.

  If the calibration data has a clean gap, one boundary goes in the middle of the gap. Reported rates are always out-of-fold.
- **Paired tests** (`pipeline.compare`): the difference between two generators, bootstrapped over the same test images and pooled over seeds.

## 6. Explanations

`xai/odam.py` hooks the three feature maps entering YOLO26's head and backpropagates one detection's score. The map is `ReLU(sum over channels of grad x activation)`, kept per location, which is why it is specific to one detection (plain Grad-CAM averages the gradient away).

`xai/drise.py` only calls `predict`, so it explains any runtime, including INT8. Hidden pixels become the image's mean colour. A light blur was tried first, and the test showed large defects survived it, giving a flat map.

`xai/study.py` runs four experiments:

1. **ODAM localisation per training condition**, scored with the pointing game and energy-on-defect against MVTec's pixel masks.
2. **Faithfulness** (deletion and insertion), ODAM vs D-RISE.
3. **FP32 vs INT8 D-RISE agreement**, using identical masks.
4. **The explanation gate**: does "share of the explanation inside the detection's own box" separate true from false detections, and does routing low-share rejections to review help?

The gate is a hypothesis. Report what the numbers say, including "it did not help".

## 7. What to Be Ready to Explain in an Interview

- Why the first third of each defect type, and what goes wrong if you shuffle.
- Why the mask bank removes DefectFill's leak, and what that costs DefectFill.
- Why LoRA instead of DreamBooth for MAGIC, and how that could bias the comparison against it.
- What Gaussian Prompt Perturbation does, and why MAGIC applies it at training time too.
- Why the last checkpoint and not the best.
- How the review band's thresholds are chosen, and why cross-fitting matters.
- Why plain Grad-CAM is wrong for detectors, and what ODAM changes.
- Why D-RISE can explain an INT8 model and ODAM cannot.
- Why the detection head stays FP32 in INT8.
- What NMS-free means and how it simplified the browser demo.
- Why AGPL matters if Bosch wanted to ship this, and what you would swap (RF-DETR is Apache 2.0).
- What PSI measures, and what you would do when it crosses 0.25.
