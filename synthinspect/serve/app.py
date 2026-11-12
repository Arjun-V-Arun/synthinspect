"""Inspection API: the research model behind a small, observable service.

POST /v1/inspect   image -> pass / review / reject, score, boxes, model version
POST /v1/explain   image -> D-RISE heatmap for the top detection (black-box, so it explains the INT8 model)
GET  /v1/model     model card: training data, metrics, thresholds, version hash
GET  /healthz /readyz /metrics

The model folder (made by `python -m synthinspect demo`) holds config.json and
either an OpenVINO INT8 model (preferred) or an ONNX model. No PyTorch needed.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from ..detect.runtime import OnnxDetector, OpenVINODetector
from ..eval.review import Band
from ..io import read_json
from .monitoring import Monitor

DECISIONS = {0: "pass", 1: "review", 2: "reject"}
MAX_BYTES = 15 * 1024 * 1024


class Service:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.cfg = read_json(self.dir / "config.json")
        imgsz, fmt = self.cfg["imgsz"], self.cfg["format"]
        threads = int(os.environ.get("SYNTHINSPECT_THREADS", "0")) or None
        ov_path = self.dir / "openvino" / "int8" / "model.xml"
        if ov_path.exists() and os.environ.get("SYNTHINSPECT_RUNTIME", "openvino") == "openvino":
            self.det = OpenVINODetector(ov_path, imgsz, fmt, threads=threads)
            self.runtime = "openvino-int8"
        else:
            self.det = OnnxDetector(self.dir / "model.onnx", imgsz, fmt, threads=threads)
            self.runtime = "onnxruntime-fp32"
        self.band = Band(**self.cfg["band"])
        self.version = self.cfg["version"]
        self.monitor = Monitor(self.cfg.get("reference_scores"))

    def inspect(self, img: np.ndarray) -> dict:
        t = time.perf_counter()
        dets = self.det.predict([img], min_score=0.05)[0].top(20)
        ms = (time.perf_counter() - t) * 1000
        score = dets.image_score
        decision = DECISIONS[int(self.band.decide(np.array([score]))[0])]
        self.monitor.record(decision, score, ms)
        return {
            "decision": decision,
            "score": round(score, 4),
            "detections": dets.to_dict(),
            "thresholds": self.band.to_dict(),
            "model_version": self.version,
            "runtime": self.runtime,
            "latency_ms": round(ms, 1),
        }


app = FastAPI(title="SynthInspect", version="0.1.0")
_service: Service | None = None


def service() -> Service:
    global _service
    if _service is None:
        d = os.environ.get("SYNTHINSPECT_MODEL_DIR")
        if not d:
            raise HTTPException(503, "SYNTHINSPECT_MODEL_DIR is not set")
        _service = Service(d)
    return _service


async def _image(file: UploadFile) -> np.ndarray:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 15 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a readable image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    s = service()
    return {"ready": True, "runtime": s.runtime, "model_version": s.version}


@app.get("/v1/model")
def model_card():
    s = service()
    keep = ("category", "condition", "detector", "format", "imgsz", "band", "version", "metrics", "export", "license")
    return {k: s.cfg[k] for k in keep if k in s.cfg} | {"runtime": s.runtime}


@app.post("/v1/inspect")
async def inspect(file: UploadFile = File(...)):
    return service().inspect(await _image(file))


@app.post("/v1/explain")
async def explain(file: UploadFile = File(...), n_masks: int = Query(200, ge=50, le=2000)):
    from ..xai.drise import drise

    s = service()
    img = await _image(file)
    top = s.det.predict([img], min_score=0.0)[0].top(1)
    if len(top) == 0:
        raise HTTPException(422, "Nothing detected to explain")
    t = time.perf_counter()
    sal = drise(s.det, img, top.boxes[0], n_masks=n_masks, grid=12, batch=16)
    heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)
    ok, png = cv2.imencode(".png", overlay)
    return {
        "box": top.boxes[0].round(1).tolist(),
        "score": round(float(top.scores[0]), 4),
        "method": f"D-RISE ({n_masks} masks, black-box on {s.runtime})",
        "seconds": round(time.perf_counter() - t, 2),
        "overlay_png_base64": base64.b64encode(png.tobytes()).decode("ascii"),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    s = service()
    return s.monitor.prometheus(s.version)
