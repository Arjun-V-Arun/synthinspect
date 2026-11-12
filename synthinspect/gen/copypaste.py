"""Copy-paste: move a real training defect onto a new good part.

This is step one of the RGCB approach and also its own ablation (copy-paste
without style transfer). The defect's pixels move with exactly the same affine
transform as its mask, so the label stays correct by construction.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..data.maskbank import MaskBank, PlacedMask
from ..data.mvtec import Split


class CopyPasteGenerator:
    name = "copypaste"

    def __init__(self, blend: str = "poisson", feather: int = 5, context: int = 8):
        if blend not in ("poisson", "alpha"):
            raise ValueError("blend must be 'poisson' or 'alpha'")
        self.blend = blend
        self.feather = feather
        self.context = context

    def fit(self, split: Split, bank: MaskBank, workdir: Path) -> None:
        return None  # nothing to learn

    def paint(self, host: np.ndarray, placed: PlacedMask, bank: MaskBank, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        src = bank.source_image(placed.source_index)
        h, w = host.shape[:2]
        warped = cv2.warpAffine(src, placed.affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out, used = paste(host, warped, placed.mask, self.blend, self.feather, self.context)
        return out, placed.mask.copy(), {"blend": used}


def paste(host: np.ndarray, src: np.ndarray, mask: np.ndarray, blend: str, feather: int, context: int) -> tuple[np.ndarray, str]:
    """Blend ``src`` into ``host`` inside ``mask`` (both same size). Returns the image and the blend actually used."""
    m = mask.astype(np.uint8)
    if blend == "poisson":
        # OpenCV erodes the clone region internally; with a margin under ~6 px the defect's
        # edge gradient falls outside it and the defect fades away (caught by the tests).
        context = max(context, 6)
    if context > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * context + 1, 2 * context + 1))
        m = cv2.dilate(m, k)
    if blend == "poisson":
        ys, xs = np.nonzero(m)
        if len(xs):
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            h, w = m.shape
            # seamlessClone fails when the region touches the border; fall back to alpha then.
            if x1 > 1 and y1 > 1 and x2 < w - 2 and y2 < h - 2:
                center = (int((x1 + x2 + 1) // 2), int((y1 + y2 + 1) // 2))
                try:
                    out = cv2.seamlessClone(
                        cv2.cvtColor(src, cv2.COLOR_RGB2BGR), cv2.cvtColor(host, cv2.COLOR_RGB2BGR), m * 255, center, cv2.NORMAL_CLONE
                    )
                    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), "poisson"
                except cv2.error:
                    pass
    alpha = feather_alpha(m > 0, feather)[..., None]
    out = (alpha * src.astype(np.float32) + (1 - alpha) * host.astype(np.float32)).round().clip(0, 255).astype(np.uint8)
    return out, "alpha"


def feather_alpha(mask: np.ndarray, feather: int) -> np.ndarray:
    a = mask.astype(np.float32)
    if feather > 0:
        k = 2 * feather + 1
        a = cv2.GaussianBlur(a, (k, k), 0)
        a = np.maximum(a, mask.astype(np.float32))  # never soften inside the mask itself
    return a
