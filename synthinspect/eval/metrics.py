"""Image-level and box-level metrics with bootstrap confidence intervals.

Image level (the part's pass/fail): the score of a part is its most confident
detection. AUROC and AP treat "defective" as the positive class.

Box level (did it find the right place): class-agnostic AP at IoU 0.5 and at a
lenient IoU 0.1. Boxes made from defect masks are ill-defined for thin or
fragmented defects (a scratch's box is mostly background), so the lenient
threshold is reported too; inspection papers do the same (one recent study
uses IoU 0.01).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detect.base import Detections


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties counted as half."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the precision-recall step curve (same definition as scikit-learn)."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    s_sorted, y_sorted = np.asarray(s)[order], y[order]
    # group tied scores into one threshold
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[idx]
    fp = (idx + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    prev = np.r_[0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def box_ap(preds: list[Detections], gts: list[np.ndarray], iou_thr: float = 0.5, max_det: int = 100) -> float:
    """Class-agnostic VOC/COCO-style AP with all-point interpolation, pooled over images."""
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return float("nan")
    rows = []  # (score, is_tp)
    for p, g in zip(preds, gts):
        p = p.top(max_det)
        matched = np.zeros(len(g), bool)
        ious = iou_matrix(p.boxes, g)
        for i in range(len(p)):
            tp = False
            if len(g):
                cand = np.where((ious[i] >= iou_thr) & ~matched)[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
                    matched[j] = True
                    tp = True
            rows.append((p.scores[i], tp))
    if not rows:
        return 0.0
    rows.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows])
    fp = np.cumsum([not r[1] for r in rows])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    # all-point interpolation
    mrec = np.r_[0, recall, recall[-1]]
    mpre = np.r_[1, precision, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@dataclass
class EvalData:
    labels: np.ndarray  # 1 = defective part
    types: list[str]
    preds: list[Detections]
    gts: list[np.ndarray]

    @property
    def scores(self) -> np.ndarray:
        return np.array([p.image_score for p in self.preds])


def compute(d: EvalData, idx: np.ndarray | None = None) -> dict[str, float]:
    if idx is None:
        idx = np.arange(len(d.labels))
    y, s = d.labels[idx], d.scores[idx]
    preds = [d.preds[i] for i in idx]
    gts = [d.gts[i] for i in idx]
    return {
        "image_auroc": auroc(y, s),
        "image_ap": average_precision(y, s),
        "box_ap50": box_ap(preds, gts, 0.5),
        "box_ap10": box_ap(preds, gts, 0.1),
    }


def stratified_bootstrap(labels: np.ndarray, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    for _ in range(n):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def with_ci(d: EvalData, n_boot: int = 1000, seed: int = 0) -> dict[str, dict[str, float]]:
    point = compute(d)
    samples = {k: [] for k in point}
    for idx in stratified_bootstrap(d.labels, n_boot, seed):
        for k, v in compute(d, idx).items():
            samples[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(samples[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
        out[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
    return out


def paired_difference(a: EvalData, b: EvalData, metric: str, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Bootstrap the difference metric(a) - metric(b) on the SAME test images (paired)."""
    if not np.array_equal(a.labels, b.labels):
        raise ValueError("Paired comparison needs the same test images in the same order")
    diffs = []
    for idx in stratified_bootstrap(a.labels, n_boot, seed):
        diffs.append(compute(a, idx)[metric] - compute(b, idx)[metric])
    diffs = np.array(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    point = compute(a)[metric] - compute(b)[metric]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_two_sided": float(min(1.0, p))}
