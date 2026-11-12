"""Rough part/background separation, used to keep synthetic defects on the part.

This stands in for MAGIC's U2-Net foreground mask. It is deliberately simple:
estimate the background colour from the image border, threshold the colour
distance with Otsu, clean up, and keep the largest connected region. Texture
categories (the whole image is the part) use mode="full".
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_foreground(img: np.ndarray, mode: str = "auto", border: int = 8, min_frac: float = 0.02) -> np.ndarray:
    h, w = img.shape[:2]
    if mode == "full":
        return np.ones((h, w), dtype=bool)
    if mode != "auto":
        raise ValueError(f"Unknown foreground mode: {mode}")

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge = np.concatenate(
        [lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3), lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)]
    )
    bg = np.median(edge, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dist8 = cv2.GaussianBlur(dist8, (5, 5), 0)
    _, fg = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, (min(h, w) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.ones((h, w), dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = labels == largest
    keep = _fill_holes(keep)
    if keep.mean() < min_frac:  # threshold failed, fall back to the whole image
        return np.ones((h, w), dtype=bool)
    return keep


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Flood from a border pixel that is background; if none, the mask covers the border.
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for x, y in seeds:
        if flood[y, x] == 0:
            cv2.floodFill(flood, ff_mask, (x, y), 255)
            break
    else:
        return mask
    holes = cv2.bitwise_not(flood)
    return (m | holes) > 0
