"""D-RISE (Petsiuk et al., CVPR 2021): black-box saliency for one detection.

Hide random parts of the image many times, run the detector, and credit each
visible region by how well the masked image still produces the target
detection: weight = max over detections of IoU(target box, box) * score.
Because it only calls ``predict``, the same code explains the PyTorch model,
the ONNX export and the INT8 OpenVINO model, which is what lets us ask whether
quantisation changed what the model looks at.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..eval.metrics import iou_matrix


def random_masks(n: int, shape: tuple[int, int], grid: int = 16, p: float = 0.5, seed: int = 0) -> np.ndarray:
    """RISE masks: coarse random grids, upsampled smoothly and randomly shifted. (n, H, W) float32 in [0, 1]."""
    rng = np.random.default_rng(seed)
    h, w = shape
    ch, cw = int(np.ceil(h / grid)), int(np.ceil(w / grid))
    out = np.empty((n, h, w), np.float32)
    for i in range(n):
        g = (rng.random((grid, grid)) < p).astype(np.float32)
        up = cv2.resize(g, ((grid + 1) * cw, (grid + 1) * ch), interpolation=cv2.INTER_LINEAR)
        dx, dy = rng.integers(0, cw), rng.integers(0, ch)
        out[i] = up[dy : dy + h, dx : dx + w]
    return out


def apply_mask(image: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    m = mask[..., None]
    return (image.astype(np.float32) * m + background.astype(np.float32) * (1 - m)).astype(np.uint8)


def target_weight(dets, box: np.ndarray, cls: int | None = None) -> float:
    if len(dets) == 0:
        return 0.0
    keep = np.ones(len(dets), bool) if cls is None else dets.classes == cls
    if not keep.any():
        return 0.0
    ious = iou_matrix(box[None], dets.boxes[keep])[0]
    return float((ious * dets.scores[keep]).max())


def drise(
    det,
    image: np.ndarray,
    box: np.ndarray,
    cls: int | None = None,
    n_masks: int = 1000,
    grid: int = 16,
    p: float = 0.5,
    batch: int = 16,
    seed: int = 0,
    fill: str = "mean",
) -> np.ndarray:
    """Saliency (original image size, [0, 1]) for the detection ``box``.

    The masked images go through the detector's own preprocessing, so the
    explanation covers the whole pipeline. Hidden pixels become the image's mean
    colour ("mean"), or a heavily blurred copy ("blur"). A light blur is not
    enough: large defects survive it and the map comes out flat (a test caught this).
    """
    h, w = image.shape[:2]
    masks = random_masks(n_masks, (h, w), grid, p, seed)
    if fill == "mean":
        background = np.broadcast_to(image.reshape(-1, 3).mean(0).astype(np.uint8), image.shape)
    elif fill == "blur":
        background = cv2.GaussianBlur(image, (0, 0), sigmaX=max(h, w) / 10)
    else:
        raise ValueError(fill)
    weights = np.zeros(n_masks, np.float32)
    for i in range(0, n_masks, batch):
        imgs = [apply_mask(image, m, background) for m in masks[i : i + batch]]
        preds = det.predict(imgs, min_score=0.01)
        weights[i : i + len(imgs)] = [target_weight(d, box, cls) for d in preds]
    sal = np.tensordot(weights, masks, axes=1) / (n_masks * p)
    sal = sal - sal.min()
    return (sal / sal.max()).astype(np.float32) if sal.max() > 0 else sal.astype(np.float32)
