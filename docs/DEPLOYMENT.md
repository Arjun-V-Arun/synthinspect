# Shipping Small and Moving to Production

This covers how the trained model gets small, where the demo lives, and what a real production rollout would add. Commands assume Windows PowerShell from the repo root.

## 1. Shipping Small

`python -m synthinspect export --category metal_nut` takes one trained run and produces every runtime from the same weights:

| Artifact | What it is | Used by |
|---|---|---|
| `model.onnx` | FP32, NMS-free YOLO26 head, output `(1, 300, 6)` | baseline, API fallback |
| `openvino/fp16/model.xml` | FP16 weights | CPU/iGPU servers |
| `openvino/int8/model.xml` | INT8 post-training quantisation (NNCF), head kept FP32 | **the API** |
| `model.int8.onnx` | INT8 QDQ for ONNX Runtime | **the browser demo** (WASM) |
| `model.fp16.onnx` | FP16 weights, FP32 head and I/O | browser fallback |

Rules that make the numbers trustworthy:

- **Calibration uses training images only** (good parts, the few real defects, the synthetic defects). The test set never touches quantisation.
- **The detection head stays in floating point.** This follows Ultralytics' own OpenVINO exporter: the box decoder and score sigmoid lose accuracy when quantised and hold almost none of the compute.
- **Every artifact is re-scored on the full test set**, so `report.json` shows exactly what INT8 costs in AP, next to its size and p50/p95 latency on your CPU.
- **Release gate:** INT8 may lose at most 0.02 image AP versus FP32 (`export.int8_max_ap_drop` in the config). If it fails, ship FP16 and record why.
- **Explanations are checked too:** `explain` runs D-RISE with identical masks on the FP32 and INT8 models and reports how much the maps agree. A quantised model that scores the same but looks somewhere else is a regression the AP number would hide.

Why YOLO26 makes this easy: its head is NMS-free, so the exported graph already returns final boxes. There is no NMS to re-implement in the API or in JavaScript; the browser decodes results in about 20 lines (`web/inspect.js`).

## 2. The Demo: In the Browser, Free, Always On

Since July 2026, Hugging Face no longer offers free Gradio or Docker Spaces on CPU Basic to new users (Static Spaces stay free). Rather than depend on that, the demo runs the model **in the visitor's browser** with ONNX Runtime Web (WASM), from a static site on GitHub Pages:

- No server, no cost, no cold start, and it is a direct demonstration of edge deployment.
- The JavaScript (`web/inspect.js`) is tested against the Python pipeline under Node with the same WASM runtime (`tests/test_web.py`): same scores, same boxes, same letterbox geometry, same pass/review/reject decision.
- Live explanations run D-RISE in the browser (120 masked runs); the saved explanations are ODAM maps made when the demo was packaged.

**Publish it:**

1. `python -m synthinspect demo --category metal_nut` (writes `web/assets/`).
2. Commit `web/assets/` (model about 3 MB plus 8 sample images).
3. GitHub repo → Settings → Pages → Source: **GitHub Actions**.
4. Push to `main`; `.github/workflows/pages.yml` publishes `web/` to `https://arjun-v-arun.github.io/<repo>/`.

**Licence note:** the sample images and the model derive from MVTec AD (CC BY-NC-SA 4.0). A non-commercial portfolio demo with attribution is within those terms; the page footer carries the attribution.

## 3. The API: From Research Code to a Service

`synthinspect/serve/app.py` wraps the INT8 model in FastAPI. The Docker image contains no PyTorch (it runs OpenVINO, with ONNX Runtime as fallback); this was checked by running the API from exactly the files the Dockerfile copies, in a clean environment.

| Endpoint | Purpose |
|---|---|
| `POST /v1/inspect` | image → `pass` / `review` / `reject`, score, boxes, thresholds, model version, latency |
| `POST /v1/explain` | D-RISE heatmap of the top detection, computed on the deployed INT8 model |
| `GET /v1/model` | model card: training data, metrics, thresholds, version hash, licence |
| `GET /healthz`, `/readyz` | liveness and readiness |
| `GET /metrics` | Prometheus: decisions, latency histogram, review rate, score drift (PSI) |

**Run locally:**

```powershell
python -m synthinspect serve --model-dir runs/metal_nut/package
# then open http://localhost:8000/docs
```

**Deploy to Google Cloud Run** (free tier: 2M requests, 240,000 vCPU-seconds and 450,000 GiB-seconds a month; scales to zero):

```powershell
gcloud auth login
gcloud config set project <your-project>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
# The Dockerfile reads the packaged model from MODEL_DIR; edit its default or pass it in cloudbuild.
gcloud run deploy synthinspect --source . --region asia-south1 --memory 1Gi --cpu 1 --max-instances 1 --allow-unauthenticated
```

`--max-instances 1` caps cost. Cloud Run needs a billing account even inside the free tier; set a budget alert.

## 4. What Production Would Add

The repo implements the parts marked **done**; the rest is the plan you can talk through in an interview.

| Stage | What happens | Status |
|---|---|---|
| Data contract | Fixed split rule, train/test disjointness asserted, masks only from training defects | done |
| Reproducible runs | Config-driven, resumable stages, seeds, per-image generation seeds | done |
| Evaluation | Bootstrap CIs, three seeds, paired tests between generators | done |
| Release gate | INT8 accuracy drop limit; explanation agreement FP32 vs INT8 | done |
| Model card + version | `config.json` with metrics, thresholds, licence, SHA-256 version | done |
| Serving | Small container, health/readiness, input limits, Prometheus metrics | done |
| Drift monitoring | PSI of live scores vs the release distribution | done (metric only, no alerting) |
| CI | Every stage tested on CPU in GitHub Actions with tiny fixtures | done |
| Shadow mode | New model scores live parts next to the current one; decisions still come from the current one; compare disagreement before switching | plan |
| Canary | Route a small share of a line to the new model; watch escape and false-reject rates against the review queue | plan |
| Feedback loop | Every part sent to review gets a human label; those labels become the next calibration set and new training defects | plan |
| Threshold calibration | On the line, thresholds come from a held-out calibration batch of that line's parts, never from the test set | plan (the demo's thresholds are for display only) |
| Retraining trigger | PSI above 0.25, or review rate drifting from its release value | plan |
