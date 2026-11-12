"""YOLO26 training (pretrained or from scratch) and a PyTorch detector with the shared preprocessing."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .base import Detections, decode_end2end, letterbox, to_nchw

os.environ.setdefault("YOLO_OFFLINE", "1")


def _quiet_ultralytics() -> None:
    from ultralytics import settings

    settings.update({"sync": False})  # no analytics


def train_yolo(
    data_yaml: str | Path,
    out_dir: str | Path,
    model: str = "yolo26n",
    pretrained: bool = True,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    seed: int = 0,
    workers: int = 2,
    device: str | None = None,
    aug: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Train for a fixed number of epochs and return the LAST checkpoint (no selection on any validation set).

    ``pretrained=False`` builds the network from its YAML with random weights (from scratch).
    """
    from ultralytics import YOLO

    _quiet_ultralytics()
    out_dir = Path(out_dir)
    weights = f"{model}.pt" if pretrained else f"{model}.yaml"
    net = YOLO(weights)
    args = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=True,
        workers=workers,
        project=str(out_dir.parent.resolve()),
        name=out_dir.name,
        exist_ok=True,
        val=False,  # never look at a validation score while training
        plots=False,
        pretrained=pretrained,
        verbose=False,
    )
    if device is not None:
        args["device"] = device
    args.update(aug or {})
    args.update(extra or {})
    net.train(**args)
    last = out_dir / "weights" / "last.pt"
    if not last.exists():
        raise FileNotFoundError(f"Training finished but {last} is missing")
    return last


class TorchYOLO:
    """PyTorch YOLO26 with the shared letterbox; also exposes the network for gradient explanations."""

    def __init__(self, weights: str | Path, imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO

        _quiet_ultralytics()
        self.name = f"torch:{Path(weights).name}"
        self.imgsz = imgsz
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = YOLO(str(weights)).model.to(self.device).float().eval()
        head = self.net.model[-1]
        if getattr(head, "one2one_cv2", None) is None:
            raise ValueError("Expected a YOLO26-style end-to-end head (one2one branch)")
        head.end2end = True  # NMS-free one-to-one predictions: (B, max_det, 6)
        self.head = head
        self.names = self.net.names

    def preprocess(self, images: list[np.ndarray]):
        boxed, infos = zip(*(letterbox(im, self.imgsz) for im in images))
        return torch.from_numpy(to_nchw(list(boxed))).to(self.device), list(infos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return y[0] if isinstance(y, (list, tuple)) else y

    @torch.no_grad()
    def predict(self, images: list[np.ndarray], min_score: float = 0.001) -> list[Detections]:
        x, infos = self.preprocess(images)
        raw = self.forward(x).float().cpu().numpy()
        return decode_end2end(raw, infos, [im.shape[:2] for im in images], min_score)
