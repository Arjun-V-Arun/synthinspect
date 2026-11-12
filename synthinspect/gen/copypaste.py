"""Copy-paste: move a real training defect onto a new good part.

This is step one of the RGCB approach and also its own ablation (copy-paste
without style transfer). The defect's pixels move with exactly the same affine
transform as its mask, so the label stays correct by construction.

Blends:
  radial  : a square patch around the defect, faded with a Gaussian alpha
            exp(-(x^2 + y^2) / (2 sigma^2)) over [-1, 1]^2, sigma 0.8. This is how the
            RGCB generator pastes colony patches onto empty dishes (lib.gaussian_alpha).
  poisson : OpenCV seamless cloning of the region around the defect.
  alpha   : feathered mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..data.maskbank import MaskBank, PlacedMask
from ..data.mvtec import Split


class CopyPasteGenerator:
    name = "copypaste"

    def __init__(self, blend: str = "radial", feather: int = 5, context: int = 8, radial_sigma: float = 0.8, radial_pad: float = 0.25):
        if blend not in ("radial", "poisson", "alpha"):
            raise ValueError("blend must be 'radial', 'poisson' or 'alpha'")
        self.blend = blend
        self.feather = feather
        self.context = context
        self.radial_sigma = radial_sigma
        self.radial_pad = radial_pad

    def fit(self, split: Split, bank: MaskBank, workdir: Path) -> None:
        return None  # nothing to learn

    def paint(self, host: np.ndarray, placed: PlacedMask, bank: MaskBank, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        src = bank.source_image(placed.source_index)
        h, w = host.shape[:2]
        warped = cv2.warpAffine(src, placed.affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out, used = paste(host, warped, placed.mask, self.blend, self.feather, self.context, self.radial_sigma, self.radial_pad)
        return out, placed.mask.copy(), {"blend": used}


def paste(
    host: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    blend: str,
    feather: int,
    context: int,
    radial_sigma: float = 0.8,
    radial_pad: float = 0.25,
) -> tuple[np.ndarray, str]:
    """Blend ``src`` into ``host`` around ``mask`` (both same size). Returns the image and the blend actually used."""
    m = mask.astype(np.uint8)
    if blend == "poisson":
        # OpenCV erodes the clone region internally; with a margin under ~6 px the defect's
        # edge gradient falls outside it and the defect fades away (caught by the tests).
        context = max(context, 6)
    if context > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * context + 1, 2 * context + 1))
        m = cv2.dilate(m, k)
    if blend == "radial":
        ys, xs = np.nonzero(m)
        if len(xs):
            h, w = m.shape
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            # Pad the patch so the defect sits in the bright middle of the Gaussian, not on its fading rim.
            px, py = int(round(radial_pad * (x2 - x1))), int(round(radial_pad * (y2 - y1)))
            x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
            xx, yy = np.meshgrid(np.linspace(-1, 1, x2 - x1), np.linspace(-1, 1, y2 - y1))
            a = np.exp(-(xx**2 + yy**2) / (2 * radial_sigma**2))[..., None]
            out = host.copy()
            patch = a * src[y1:y2, x1:x2].astype(np.float32) + (1 - a) * host[y1:y2, x1:x2].astype(np.float32)
            out[y1:y2, x1:x2] = patch.round().clip(0, 255).astype(np.uint8)
            return out, "radial"
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
