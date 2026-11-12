# Literature Review: Few-Shot Defect Synthesis, Detectors, Explanations and Deployment

Reviewed November 2026. Every claim below links to its source in the reference list at the end. Numbers are as reported by each paper; they were measured under different protocols, so they are not directly comparable across rows.

## 1. The Question

A new part goes into production. You have a few hundred photos of good parts and a handful (about 5 to 8) of photos of each defect type. Can synthetic defect images close the gap, and which way of making them works best for a detector that must also run small and explain itself?

The project compares three ways of making defect images:

1. **The RGCB approach** (what Arjun used at RGCB in 2024, from Pawłowski, Majchrowska and Golan, *Scientific Reports* 2022): paste real objects onto clean backgrounds with a Gaussian fade, then restyle the whole image toward a real clean image with photorealistic HRNet style transfer (Li et al. 2019), warm-started from image to image. The structure, and so the labels, stay put.
2. **Two recent diffusion methods**, chosen below.

## 2. Selection Criteria for the Two Recent Methods

| Criterion | Why it matters here |
|---|---|
| Published 2025 or later | The brief asks for the latest approaches |
| Few-shot (uses only about one third of each defect type's images) | Matches the new-product scenario |
| Produces a label (mask or box) with each image | A detector cannot train on unlabelled images |
| Fits an 8 GB laptop GPU (RTX 4070) | All runs happen on Arjun's machine |
| Method specified well enough to reproduce, or code released | Results must be checkable |
| Does not need held-out test masks at generation time | Otherwise the comparison leaks test information |

## 3. Landscape (2024 to 2026)

| Method | Venue | Backbone | What is trained | Compute reported | Code and licence | Fits 8 GB? |
|---|---|---|---|---|---|---|
| AnomalyDiffusion | AAAI 2024 | Latent diffusion | Textual inversion for defect and mask | 249 h on an A100 for all of MVTec AD (SeaS's measurement; SeaS takes 73 h) | Released | Yes, slow |
| DualAnoDiff | CVPR 2025 | SD 1.5 | Dual-branch fine-tune per category | Not stated in README | Released (Gitee/GitHub) | Unclear |
| **DefectFill** | CVPR 2025 | SD 2 inpainting | LoRA rank 8 on UNet attention and text encoder, three losses | Not stated | No official code | **Yes (LoRA)** |
| SeaS | ICCV 2025 | SD 1.4 | Shared U-Net and token embeddings, plus mask branch | About 22 GB on an A100 | MIT | No |
| TF-IDG | ICCV 2025 | SD 2.1 + AnyDoor ControlNet + DINOv2-g | Nothing (training-free) | Not stated | Released; no licence file; Python 3.8 / torch 2.0 | Unclear |
| **MAGIC** | CVPR 2026 Findings | SD 2 inpainting | DreamBooth fine-tune, prompt perturbation, spatially adaptive guidance | About 1.5 h per class on an RTX A6000 (48 GB) | CC BY-NC 4.0 | Not as published; yes with LoRA |
| UniDG | arXiv, Apr 2026 | FLUX.1-Fill (12B) | LoRA on a 300K-sample defect dataset | Authors call memory use substantial | "Will be available"; no licence stated | No |
| NPI textual inversion (Relimetrics) | arXiv, Apr 2026 | Frozen latent diffusion | Textual inversion with a mask loss | Under 6 GB, about 16 h on an RTX 5080 | Not released | Yes, slow |
| OSAGEN | arXiv, Jul 2026 | SD 1.5 inpainting | Three-stage curriculum | About 2.4 h per category on an A100 | Released on acceptance | Unknown |

### Reported downstream results on MVTec AD

These come from MAGIC's own comparison table (pixel-level AP of a U-Net trained on 500 generated pairs per defect type):

| Method | Pixel AP | Classification accuracy (ResNet-34) |
|---|---|---|
| AnomalyDiffusion | 74.02 | 64.90 |
| DualAnoDiff | 76.79 | 68.50 |
| SeaS | 78.33 | 52.73 |
| MAGIC | 81.99 | 78.06 |

OSAGEN reports a higher pixel AP (88.1) under its own protocol, but its code is not out yet, so it cannot be checked.

## 4. The Two Picks

### DefectFill (CVPR 2025)

Fine-tunes an inpainting diffusion model with LoRA (rank 8, UNet attention and text encoder) so a rare token `[V*]` comes to mean "this defect". Three losses do the work:

- **Defect loss** (weight 0.5): noise-prediction loss inside the defect mask only, prompt "A photo of [V*]".
- **Object loss** (weight 0.2): random box masks over the part, prompt "A [object] with [V*]", background down-weighted.
- **Attention loss** (weight 0.05): pushes the cross-attention map of `[V*]` onto the defect mask.

At generation it makes several candidates per mask and keeps the one that differs most from the clean part (Low-Fidelity Selection, by LPIPS), so samples where the model "healed" the defect away get dropped.

**Why it was picked:** the method is fully specified (losses, weights, LoRA rank, learning rates), and LoRA training of an inpainting model fits in 8 GB.

**Known problem:** it generates on the masks of the held-out test defects. Both MAGIC and OSAGEN exclude it from their tables for this reason. This project fixes it by drawing every mask from the training defects only (Section 6).

### MAGIC (CVPR 2026 Findings)

Also an inpainting fine-tune, but it changes two things DefectFill does not:

- **Gaussian Prompt Perturbation:** adds noise (sigma 1.0) to the prompt embedding during both training and generation. The paper shows that applying it only at generation time hurts, because training and test conditions no longer match.
- **Spatially Adaptive Guidance:** classifier-free guidance at 7.5 outside the defect mask, and a cosine-scheduled, weaker scale inside it (minimum 4.0), so the defect varies while the background stays faithful.

It also places masks on the part with keypoint matching (CAMA), which keeps defects on the object.

**Why it was picked:** it is the most recent method with released code, and it reports the best downstream numbers among methods with code. Its two ideas are cheap at inference, so they can be tested on the same backbone as DefectFill.

**Ambiguity found while reading:** the text says guidance starts low in the early, noisy steps and rises to 7.5, but its schedule equation, read literally, does the opposite. This project follows the text and records the choice in `configs/generators.yaml`.

### Why Not the Others

- **SeaS:** strong and MIT-licensed, but needs about 22 GB.
- **TF-IDG:** training-free is attractive, but it depends on AnyDoor with a 1B-parameter DINOv2-g, ships no licence and pins an old environment.
- **UniDG:** a 12B FLUX backbone, and no confirmed weight release.
- **OSAGEN and the Relimetrics NPI method:** no code yet.

## 5. What Changes From the Papers, and Why

| Change | Reason |
|---|---|
| Both methods use **Stable Diffusion 1.5 inpainting** (`sd-legacy/stable-diffusion-inpainting`) instead of SD 2 inpainting | The official `stabilityai` SD 2 repos started returning 404 in November 2025. The `sd-legacy` mirror of SD 1.5 inpainting is the documented replacement for the deprecated RunwayML repo. SD 1.5's smaller text encoder also helps fit 8 GB. Using one backbone for both keeps the comparison fair. |
| MAGIC uses **LoRA** instead of full DreamBooth | Full DreamBooth of the UNet does not fit 8 GB. Rank and step count are raised to compensate; this is recorded as a deviation. |
| **One shared mask bank** for every generator, built only from training defects | Removes DefectFill's test-mask leak and makes the generators differ only in how they paint, not where. |
| MAGIC's CAMA is replaced by **part-aware placement with registration** | CAMA needs GeoAware-SC, which ships without a licence. The replacement keeps each defect at the same place on the part (aligned by centre, main axis and ECC image registration) and keeps MAGIC's own alignment score as the acceptance rule. |
| The RGCB paste fades along the defect's outline instead of a square patch | RGCB's square Gaussian patch (sigma 0.8) is still 0.2 to 0.46 opaque at its border; on textured metal that showed as straight seams along long scratches. The original is kept as `blend: radial`. |

Each of these is a reason the numbers here will not match the papers. The point of the project is a controlled comparison under one protocol, not a reproduction of their tables.

## 6. Evaluation Protocol

- **Split:** for each defect type, sort the files and take the first third for training (the AnomalyDiffusion and MAGIC convention); the rest, plus the good test images, form the test set.
- **No peeking:** detectors train for a fixed number of epochs and the last checkpoint is used. No early stopping or model selection touches the test set.
- **Thresholds:** the review band is calibrated by two-fold cross-fitting on the test set, so no image is scored with thresholds it helped choose.
- **Uncertainty:** bootstrap confidence intervals over test images, three seeds per condition, and paired bootstrap differences between generators on the same test images.

## 7. Detectors: Latest Models and Trade-Offs

| Model | Released | COCO AP (smallest useful size) | Licence | Strengths | Weaknesses for this project |
|---|---|---|---|---|---|
| **YOLO26** (Ultralytics) | Sept 2025 | 40.1 (n), 2.4M params | AGPL-3.0 | NMS-free head, no DFL (simpler export and INT8), small-target label assignment (STAL), INT8 and FP16 export | AGPL means a commercial user needs an enterprise licence |
| **RF-DETR** (Roboflow) | ICLR 2026 | 48.4 (Nano), 30.5M params | Apache 2.0 (Nano to Large) | DINOv2 backbone, recommended for small datasets and domain shift, fits 6 GB for Nano/Small fine-tuning | 12x the parameters of YOLO26n; heavier on edge CPUs; INT8 support not documented |
| YOLO12 | 2025 | 40.4 (n) | AGPL-3.0 | Attention-based YOLO | The Ultralytics port is reported to train unstably |

**Picks:** YOLO26n (the edge candidate) and RF-DETR Nano (the transformer candidate).

**Pretrained vs from scratch:**

- **YOLO26n from scratch** is a real option: it is small, and a from-scratch model has no ImageNet or COCO prior, which is sometimes argued to help on industrial textures. The experiment tests this directly.
- **RF-DETR from scratch** is not run: its value is the DINOv2 backbone, and a DETR trained from random weights on a few hundred images is not a serious baseline. The docs say this rather than hide it.
- **YOLO26-P2** (an extra high-resolution head for tiny objects) ships as a config in Ultralytics but has no pretrained weights; it is available as an optional condition for screws and scratches.

## 8. Explainability for Detectors

**Short answer to "does xAI even apply to Faster R-CNN or YOLO?":** yes, but plain Grad-CAM is the wrong tool. Grad-CAM explains one class score for one image. A detector outputs many boxes, each with its own score, so the explanation has to target one detection.

| Method | Type | Works on | Cost | Benchmark finding |
|---|---|---|---|---|
| **D-RISE** (CVPR 2021) | Black-box (random masks) | Any detector, including ONNX and INT8 | 1,000 to 5,000 forward passes | Best overall on an industrial YOLOv8 study; middle of the pack on ODExAI |
| **ODAM** (ICLR 2023) | Gradient, instance-specific | Differentiable detectors (YOLO, Faster R-CNN, DETR) | One backward pass | Instance-specific maps that separate neighbouring objects |
| **G-CAME** (ACML 2023) | Gradient with a Gaussian prior | CNN detectors | About 0.4 s | Best localization on ODExAI (YOLOX, Faster R-CNN) |
| D-CLOSE | Black-box, multi-scale | Any detector | About 70 s | Most faithful on ODExAI, but slow |

**What this project does with it:**

1. **ODAM-style maps** on the PyTorch model (fast, one backward pass through YOLO26's one-to-one head).
2. **D-RISE** on any runtime, including the INT8 model, so you can ask whether quantization changed what the model looks at.
3. **Measured, not just drawn:** MVTec has pixel-level defect masks, so explanations are scored with the pointing game and energy-in-mask (EBPG) against the true defect, plus deletion and insertion for faithfulness.
4. **Built into the decision:** a detection whose own explanation mostly falls outside its box is a candidate for human review. This is tested as a hypothesis (does in-box energy separate true from false detections?), and only kept if the numbers support it.

## 9. Hosting: What Changed in 2026

- **Hugging Face Spaces** no longer offers free Gradio or Docker Spaces on CPU Basic for new users; since July 2026 those need PRO, while Static Spaces stay free. Free personal accounts can host up to two Gradio Spaces on ZeroGPU.
- **Google Cloud Run** offers 2 million requests, 240,000 vCPU-seconds and 450,000 GiB-seconds free per month, and scales to zero.
- **GitHub Pages** hosts static sites free.

**Design consequence:** the demo runs the INT8/FP16 ONNX model in the visitor's browser (a static site, so it is free and always on), and the production path is a FastAPI container on Cloud Run. See `docs/PRODUCTION.md`.

## 10. Licences

| Component | Licence | Consequence |
|---|---|---|
| MVTec AD | CC BY-NC-SA 4.0 | Non-commercial only. Derived images (including synthetic ones) must carry the same licence and attribution. |
| Ultralytics YOLO26 | AGPL-3.0 | This repository is AGPL-3.0 too. |
| RF-DETR Nano | Apache 2.0 | No restriction. |
| SD 1.5 inpainting | CreativeML OpenRAIL-M | Use restrictions apply to generated content. |
| MAGIC official code | CC BY-NC 4.0 | Not vendored. This repo re-implements the ideas from the paper. |

## References

**Defect generation**
- RGCB pipeline: Pawłowski, Majchrowska and Golan, "Generation of microbial colonies dataset with deep learning style transfer", *Scientific Reports* 12, 5212 (2022); code <https://github.com/jarek-pawlowski/microbial-dataset-generation>
- Photorealistic style transfer: Li, Ye and Li, "High-resolution network for photorealistic style transfer", arXiv:1904.11617 (2019); code (MIT) <https://github.com/limingcv/Photorealistic-Style-Transfer>
- AnomalyDiffusion: <https://www.alphaxiv.org/abs/2312.05767>
- DualAnoDiff (CVPR 2025): <https://openaccess.thecvf.com/content/CVPR2025/papers/Jin_Dual-Interrelated_Diffusion_Model_for_Few-Shot_Anomaly_Image_Generation_CVPR_2025_paper.pdf>, code <https://github.com/yinyjin/DualAnoDiff>
- DefectFill (CVPR 2025): <https://openaccess.thecvf.com/content/CVPR2025/papers/Song_DefectFill_Realistic_Defect_Generation_with_Inpainting_Diffusion_Model_for_Visual_CVPR_2025_paper.pdf>
- SeaS (ICCV 2025): <https://arxiv.org/html/2410.14987v3>, code <https://github.com/HUST-SLOW/SeaS>
- TF-IDG (ICCV 2025): <https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Training-Free_Industrial_Defect_Generation_with_Diffusion_Models_ICCV_2025_paper.pdf>, code <https://github.com/rubymiaomiao/TF-IDG>
- MAGIC (CVPR 2026 Findings): <https://arxiv.org/html/2507.02314v4>, code <https://github.com/SpatialAILab/MAGIC-Anomaly-generation>
- UniDG: <https://arxiv.org/pdf/2604.08915>
- Few-shot defect synthesis for new product introduction (Relimetrics): <https://arxiv.org/pdf/2604.22850>
- OSAGEN: <https://arxiv.org/pdf/2607.29533>

**Detectors**
- YOLO26 architecture and benchmarks: <https://arxiv.org/html/2509.25164v1>; Ultralytics model overview: <https://arxiv.org/pdf/2510.09653>
- RF-DETR (ICLR 2026): <https://iclr.cc/virtual/2026/poster/10007257>; docs <https://rfdetr.roboflow.com/latest/>
- Detector comparison, July 2026: <https://blog.jetbrains.com/pycharm/2026/07/best-object-detection-models-for-machine-learning-in-2026/>

**Explainability**
- D-RISE: <https://arxiv.org/pdf/2006.03204>
- ODAM (ICLR 2023): <https://arxiv.org/pdf/2304.06354>
- G-CAME: <https://proceedings.mlr.press/v222/truong24a/truong24a.pdf>
- ODExAI benchmark: <https://www.cs.unb.ca/~hcao3/publications/2025-KI/ODExAI.pdf>
- Black-box explainability for industrial detection: <https://arxiv.org/pdf/2411.00818>

**Data, models and hosting**
- MVTec AD: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- SD 2 repositories returning 404: <https://discuss.huggingface.co/t/error-404-https-huggingface-co-stabilityai-stable-diffusion-2/170425>
- SD 1.5 inpainting mirror notice: <https://huggingface.co/runwayml/stable-diffusion-inpainting/raw/main/README.md>
- Hugging Face Spaces hardware and SDK rules: <https://huggingface.co/docs/hub/en/spaces-overview>
- Community report on the July 2026 change: <https://discuss.huggingface.co/t/official-community-complaint-revert-free-cpu-basic-spaces-and-remove-anti-developer-sdk-restrictions/177703>
- Cloud Run: <https://cloud.google.com/run/>
