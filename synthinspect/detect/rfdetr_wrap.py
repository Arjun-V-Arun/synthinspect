"""RF-DETR (ICLR 2026) training, PyTorch inference and ONNX export, on the same YOLO-format datasets.

Only the pretrained setting is used in the main experiment: RF-DETR's value is
its DINOv2 backbone, and a DETR trained from random weights on a few hundred
images is not a meaningful baseline. ``pretrained=False`` exists for
completeness and for offline tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Detections

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


def _cls(size: str):
    import rfdetr

    return getattr(rfdetr, SIZES[size])


def train_rfdetr(
    dataset_dir: str | Path,
    out_dir: str | Path,
    size: str = "nano",
    pretrained: bool = True,
    epochs: int = 50,
    batch: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    seed: int = 0,
    workers: int = 2,
    device: str | None = None,
) -> Path:
    """Fixed-length training; returns the final EMA weights (no checkpoint selection on any validation set)."""
    kw = {} if pretrained else {"pretrain_weights": None}
    if device:
        kw["device"] = device
    model = _cls(size)(**kw)
    out_dir = Path(out_dir)
    model.train(
        dataset_dir=str(dataset_dir),
        dataset_file="yolo",
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=grad_accum,
        lr=lr,
        output_dir=str(out_dir),
        seed=seed,
        num_workers=workers,
        early_stopping=False,
        tensorboard=False,
        progress_bar=None,
        run_test=False,
    )
    for name in ("last_ema.pth", "last.pth", "checkpoint.pth"):
        if (out_dir / name).exists():
            return out_dir / name
    raise FileNotFoundError(f"No final checkpoint in {out_dir}")


class TorchRFDETR:
    def __init__(self, checkpoint: str | Path, device: str | None = None, num_classes: int = 1):
        from rfdetr.detr import RFDETR

        kw = {"device": device} if device else {}
        self.model = RFDETR.from_checkpoint(str(checkpoint), trust_checkpoint=True, **kw)
        self.imgsz = int(self.model.model_config.resolution)
        self.name = f"torch:{Path(checkpoint).name}"
        self.num_classes = num_classes

    def predict(self, images: list[np.ndarray], min_score: float = 0.001) -> list[Detections]:
        res = self.model.predict(list(images), threshold=min_score, include_source_image=False)
        res = res if isinstance(res, list) else [res]
        out = []
        for r in res:
            cls = np.asarray(r.class_id, np.int64)
            keep = cls < self.num_classes  # drop the extra no-class logit slot (see runtime.decode)
            out.append(Detections(np.asarray(r.xyxy, np.float32).reshape(-1, 4)[keep], np.asarray(r.confidence, np.float32)[keep], cls[keep]))
        return out


def export_rfdetr_onnx(checkpoint: str | Path, out_path: str | Path) -> tuple[Path, int]:
    """Export to ONNX (outputs "dets" and "labels"); returns (path, input resolution)."""
    from rfdetr.detr import RFDETR

    model = RFDETR.from_checkpoint(str(checkpoint), trust_checkpoint=True, device="cpu")
    out_path = Path(out_path)
    produced = Path(model.export(output_dir=str(out_path.parent / "_rfdetr_export"), verbose=False))
    if produced.is_dir():
        produced = next(produced.glob("*.onnx"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    produced.replace(out_path)
    return out_path, int(model.model_config.resolution)
