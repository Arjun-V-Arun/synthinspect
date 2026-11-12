"""One detection format and one preprocessing path for every runtime.

PyTorch, ONNX Runtime, OpenVINO (FP32/FP16/INT8) and the browser demo all use
the same letterbox, so their outputs can be compared box for box. Boxes are
always returned in original image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass
class Detections:
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # x1 y1 x2 y2, original pixels
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))

    def __len__(self) -> int:
        return len(self.scores)

    def top(self, k: int | None = None, min_score: float = 0.0) -> "Detections":
        keep = np.argsort(-self.scores)
        keep = keep[self.scores[keep] >= min_score][:k]
        return Detections(self.boxes[keep], self.scores[keep], self.classes[keep])

    @property
    def image_score(self) -> float:
        """Part-level defect score: the most confident detection, 0 if none."""
        return float(self.scores.max()) if len(self) else 0.0

    def to_dict(self) -> dict:
        return {"boxes": self.boxes.round(1).tolist(), "scores": self.scores.round(4).tolist(), "classes": self.classes.tolist()}


@dataclass
class Letterbox:
    ratio: float
    pad_x: float
    pad_y: float
    size: int

    def to_original(self, boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        b = boxes.astype(np.float32).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / self.ratio
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / self.ratio
        h, w = shape
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        return b


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to size x size (centered), like Ultralytics."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return out, Letterbox(r, left, top, size)


def to_nchw(imgs: list[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.stack(imgs).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


def decode_end2end(raw: np.ndarray, boxes_info: list[Letterbox], shapes: list[tuple[int, int]], min_score: float) -> list[Detections]:
    """YOLO26 NMS-free output (B, max_det, 6) = x1 y1 x2 y2 score class, in letterboxed pixels."""
    out = []
    for row, lb, shape in zip(raw, boxes_info, shapes):
        keep = row[:, 4] >= min_score
        r = row[keep]
        out.append(Detections(lb.to_original(r[:, :4], shape), r[:, 4].astype(np.float32), r[:, 5].astype(np.int64)))
    return out


class Detector(Protocol):
    name: str
    imgsz: int

    def predict(self, images: list[np.ndarray], min_score: float = 0.001) -> list[Detections]: ...


def predict_all(det: Detector, images: list[np.ndarray], batch: int = 8, min_score: float = 0.001) -> list[Detections]:
    out: list[Detections] = []
    for i in range(0, len(images), batch):
        out.extend(det.predict(images[i : i + batch], min_score=min_score))
    return out
