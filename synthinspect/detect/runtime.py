"""Exported-model detectors: ONNX Runtime and OpenVINO (FP32, FP16 or INT8), same interface as TorchYOLO.

Only the decoder differs between model families:
- "yolo26": one output (B, max_det, 6) = x1 y1 x2 y2 score class in letterboxed pixels (NMS-free).
- "rfdetr": outputs "dets" (B, Q, 4) as normalised cx cy w h and "labels" (B, Q, C) logits.
  RF-DETR resizes (no letterbox) to its square resolution and normalises with ImageNet statistics.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import Detections, Letterbox, decode_end2end, letterbox, to_nchw

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def preprocess(images: list[np.ndarray], imgsz: int, fmt: str) -> tuple[np.ndarray, list[Letterbox]]:
    if fmt == "yolo26":
        boxed, infos = zip(*(letterbox(im, imgsz) for im in images))
        return to_nchw(list(boxed)), list(infos)
    if fmt == "rfdetr":
        x = to_nchw([cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR) for im in images])
        return (x - _IMAGENET_MEAN) / _IMAGENET_STD, [Letterbox(1.0, 0, 0, imgsz) for _ in images]
    raise ValueError(f"Unknown model format {fmt!r}")


def decode(
    outputs: dict[str, np.ndarray], infos: list[Letterbox], shapes: list[tuple[int, int]], fmt: str, min_score: float, num_classes: int = 1
) -> list[Detections]:
    if fmt == "yolo26":
        raw = next(iter(outputs.values()))
        return decode_end2end(raw, infos, shapes, min_score)
    dets, logits = outputs["dets"], outputs["labels"]
    # A fine-tuned RF-DETR has one extra logit slot beyond the real classes; it is not a class (found by
    # comparing against rfdetr's own predict, which reports it as an "unmapped" class id).
    probs = 1.0 / (1.0 + np.exp(-logits[..., :num_classes]))
    out = []
    for d, p, (h, w) in zip(dets, probs, shapes):
        cls = p.argmax(-1)
        score = p.max(-1)
        keep = score >= min_score
        cx, cy, bw, bh = d[keep].T
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        out.append(Detections(boxes.astype(np.float32), score[keep].astype(np.float32), cls[keep].astype(np.int64)))
    return out


class OnnxDetector:
    def __init__(self, path: str | Path, imgsz: int, fmt: str = "yolo26", threads: int | None = None, num_classes: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.name, self.imgsz, self.fmt, self.num_classes = f"onnx:{Path(path).name}", imgsz, fmt, num_classes

    def raw(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return dict(zip(self.outputs, self.session.run(None, {self.input: x})))

    def predict(self, images: list[np.ndarray], min_score: float = 0.001) -> list[Detections]:
        out = []
        for im in images:  # exported with a fixed batch of 1
            x, infos = preprocess([im], self.imgsz, self.fmt)
            out.extend(decode(self.raw(x), infos, [im.shape[:2]], self.fmt, min_score, self.num_classes))
        return out


class OpenVINODetector:
    def __init__(self, path: str | Path, imgsz: int, fmt: str = "yolo26", device: str = "CPU", threads: int | None = None, num_classes: int = 1):
        import openvino as ov

        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = threads
        self.compiled = core.compile_model(core.read_model(str(path)), device, cfg)
        self.outputs = [o.get_any_name() for o in self.compiled.outputs]
        self.name, self.imgsz, self.fmt, self.num_classes = f"openvino:{Path(path).parent.name}/{Path(path).name}", imgsz, fmt, num_classes

    def raw(self, x: np.ndarray) -> dict[str, np.ndarray]:
        res = self.compiled([x])
        return {name: res[o] for name, o in zip(self.outputs, self.compiled.outputs)}

    def predict(self, images: list[np.ndarray], min_score: float = 0.001) -> list[Detections]:
        out = []
        for im in images:
            x, infos = preprocess([im], self.imgsz, self.fmt)
            out.extend(decode(self.raw(x), infos, [im.shape[:2]], self.fmt, min_score, self.num_classes))
        return out


def load_detector(path: str | Path, imgsz: int, fmt: str = "yolo26", threads: int | None = None, num_classes: int = 1):
    """Pick the runtime from the file: .pt -> PyTorch, .onnx -> ONNX Runtime, .xml -> OpenVINO."""
    path = Path(path)
    if path.suffix == ".pt":
        from .yolo import TorchYOLO

        return TorchYOLO(path, imgsz)
    if path.suffix == ".pth":
        from .rfdetr_wrap import TorchRFDETR

        return TorchRFDETR(path, num_classes=num_classes)
    if path.suffix == ".onnx":
        return OnnxDetector(path, imgsz, fmt, threads, num_classes)
    if path.suffix == ".xml":
        return OpenVINODetector(path, imgsz, fmt, threads=threads, num_classes=num_classes)
    raise ValueError(f"Don't know how to load {path}")
