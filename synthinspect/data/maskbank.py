"""Where synthetic defects go: one shared mask source for every generator.

Masks come only from the few training defects (never from test images), which
removes DefectFill's test-mask leak and makes every generator paint into the
same kind of region. Each sampled mask is a training mask moved onto a new
good part with a random flip, rotation and scale, and accepted only if most of
it lands on the part. The acceptance score is MAGIC's alignment score,
|M_a AND M_fg| / |M_a| (their CAMA module), with a simpler foreground estimate
instead of keypoint matching.

The affine transform is returned with the mask so copy-paste can move the
defect's pixels with exactly the same geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..io import read_mask, read_rgb
from .foreground import estimate_foreground
from .mvtec import Sample


@dataclass
class PlacementPolicy:
    rotate: str = "free"  # "free" (any angle), "small" (+-15 deg), "none"
    flip: bool = True
    scale: tuple[float, float] = (0.8, 1.25)
    min_alignment: float = 0.9
    max_tries: int = 50
    foreground: str = "auto"  # "auto" or "full" (texture categories)
    keep_position: bool = False  # True for fixed-layout parts: jitter around the original location only
    position_jitter: float = 0.1  # fraction of image size, used when keep_position is True


@dataclass
class PlacedMask:
    mask: np.ndarray  # bool, host image size
    source_index: int  # index into MaskBank.sources
    affine: np.ndarray  # 2x3, source pixel coords -> host pixel coords
    alignment: float
    defect_type: str


class MaskBank:
    def __init__(self, defects: list[Sample], policy: PlacementPolicy | None = None):
        if not defects:
            raise ValueError("MaskBank needs at least one training defect")
        self.policy = policy or PlacementPolicy()
        self.sources = list(defects)
        self._masks: dict[int, np.ndarray] = {}
        self._images: dict[int, np.ndarray] = {}
        self.by_type: dict[str, list[int]] = {}
        for i, s in enumerate(self.sources):
            if not s.is_defect or s.mask is None:
                raise ValueError(f"MaskBank source is not a labelled defect: {s.image}")
            self.by_type.setdefault(s.defect_type, []).append(i)

    def source_mask(self, i: int) -> np.ndarray:
        if i not in self._masks:
            self._masks[i] = read_mask(self.sources[i].mask)
        return self._masks[i]

    def source_image(self, i: int) -> np.ndarray:
        if i not in self._images:
            self._images[i] = read_rgb(self.sources[i].image)
        return self._images[i]

    def sample(
        self,
        host: np.ndarray,
        defect_type: str,
        rng: np.random.Generator,
        host_fg: np.ndarray | None = None,
    ) -> PlacedMask:
        if defect_type not in self.by_type:
            raise KeyError(f"No training defects of type {defect_type!r}")
        p = self.policy
        h, w = host.shape[:2]
        fg = host_fg if host_fg is not None else estimate_foreground(host, p.foreground)
        fg_points = np.argwhere(fg)
        if len(fg_points) == 0:
            fg_points = np.argwhere(np.ones_like(fg))

        best: PlacedMask | None = None
        for _ in range(p.max_tries):
            idx = int(rng.choice(self.by_type[defect_type]))
            src_mask = self.source_mask(idx)
            sh, sw = src_mask.shape
            ys, xs = np.nonzero(src_mask)
            if len(xs) == 0:
                continue
            cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0

            angle = {"free": rng.uniform(0, 360), "small": rng.uniform(-15, 15), "none": 0.0}[p.rotate]
            scale = rng.uniform(*p.scale)
            fx = -1.0 if (p.flip and rng.random() < 0.5) else 1.0
            fy = -1.0 if (p.flip and rng.random() < 0.5) else 1.0

            if p.keep_position:
                tx = cx * w / sw + rng.uniform(-p.position_jitter, p.position_jitter) * w
                ty = cy * h / sh + rng.uniform(-p.position_jitter, p.position_jitter) * h
            else:
                ty, tx = fg_points[rng.integers(len(fg_points))]
                tx, ty = float(tx), float(ty)

            affine = _affine(cx, cy, tx, ty, angle, scale, fx, fy)
            placed = cv2.warpAffine(src_mask.astype(np.uint8), affine, (w, h), flags=cv2.INTER_NEAREST) > 0
            area = placed.sum()
            if area < 4:
                continue
            align = float((placed & fg).sum() / area)
            cand = PlacedMask(placed, idx, affine, align, defect_type)
            if best is None or align > best.alignment:
                best = cand
            if align >= p.min_alignment:
                return cand
        if best is None:
            raise RuntimeError(f"Could not place any {defect_type} mask")
        return best


def _affine(cx: float, cy: float, tx: float, ty: float, angle_deg: float, scale: float, fx: float, fy: float) -> np.ndarray:
    """2x3 matrix: move (cx, cy) to the origin, flip, rotate and scale, then move to (tx, ty)."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * scale
    lin = rot @ np.diag([fx, fy])
    t = np.array([tx, ty]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]]).astype(np.float64)
