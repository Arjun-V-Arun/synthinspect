"""Scoring explanations, so "the heatmap looks right" becomes a number.

Localisation (needs the true defect mask, which MVTec provides per pixel):
  pointing game : is the hottest pixel on the defect (with a small tolerance)?
  energy in mask: share of the saliency mass that falls on the defect (EBPG)
Faithfulness (needs only the model):
  deletion  : remove the most salient pixels first; a faithful map makes the
              detection score collapse fast (lower area under the curve is better)
  insertion : start from a blurred image and add the most salient pixels first;
              the score should recover fast (higher area is better)
"""

from __future__ import annotations

import cv2
import numpy as np

from .drise import target_weight


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def pointing_game(sal: np.ndarray, gt_mask: np.ndarray, tolerance: int = 15) -> bool:
    y, x = np.unravel_index(int(np.argmax(sal)), sal.shape)
    return bool(_dilate(gt_mask, tolerance)[y, x])


def energy_in(sal: np.ndarray, region: np.ndarray) -> float:
    s = np.maximum(sal, 0)
    total = s.sum()
    return float(s[region.astype(bool)].sum() / total) if total > 0 else 0.0


def box_region(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    m[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = True
    return m


def _order(sal: np.ndarray, block: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = sal.shape
    gh, gw = int(np.ceil(h / block)), int(np.ceil(w / block))
    small = cv2.resize(sal, (gw, gh), interpolation=cv2.INTER_AREA)
    return np.argsort(-small.ravel(), kind="stable"), (gh, gw)


def deletion_insertion(det, image: np.ndarray, box: np.ndarray, sal: np.ndarray, steps: int = 10, block: int = 16, cls: int | None = None) -> dict[str, float]:
    """Areas under the deletion and insertion curves (score normalised by the unperturbed score)."""
    h, w = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=max(h, w) / 50)
    order, (gh, gw) = _order(sal, block)
    base = target_weight(det.predict([image], min_score=0.01)[0], box, cls)
    if base <= 0:
        return {"deletion": float("nan"), "insertion": float("nan"), "base": 0.0}
    fracs = np.linspace(0, 1, steps + 1)
    del_imgs, ins_imgs = [], []
    for f in fracs:
        keep = np.zeros(gh * gw, np.float32)
        keep[order[: int(round(f * gh * gw))]] = 1
        m = cv2.resize(keep.reshape(gh, gw), (w, h), interpolation=cv2.INTER_NEAREST)[..., None]
        del_imgs.append((image * (1 - m) + blurred * m).astype(np.uint8))
        ins_imgs.append((blurred * (1 - m) + image * m).astype(np.uint8))
    d = [target_weight(p, box, cls) / base for p in det.predict(del_imgs, min_score=0.01)]
    i = [target_weight(p, box, cls) / base for p in det.predict(ins_imgs, min_score=0.01)]
    return {"deletion": float(np.trapezoid(d, fracs)), "insertion": float(np.trapezoid(i, fracs)), "base": base}


def saliency_agreement(a: np.ndarray, b: np.ndarray, size: int = 64, top: float = 0.1) -> dict[str, float]:
    """How similar two maps are (e.g. FP32 vs INT8): rank correlation and overlap of their hottest 10%."""
    from scipy.stats import spearmanr

    sa = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).ravel()
    sb = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA).ravel()
    k = max(1, int(top * len(sa)))
    ta, tb = set(np.argsort(-sa)[:k]), set(np.argsort(-sb)[:k])
    rho = spearmanr(sa, sb).statistic if sa.std() > 0 and sb.std() > 0 else float("nan")
    return {"spearman": float(rho), "top10_iou": len(ta & tb) / len(ta | tb)}
