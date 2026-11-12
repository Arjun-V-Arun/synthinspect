"""Where synthetic defects go: one shared mask source for every generator.

Masks come only from the few training defects (never from test images), which
removes DefectFill's test-mask leak and makes every generator paint into the
same kind of region. Each sampled mask is a training mask moved onto a new
good part, and accepted only if most of it lands on the part (MAGIC's
alignment score, |M_a AND M_fg| / |M_a|).

How the mask moves depends on the part, set per category in the config:

- "aligned" (metal_nut, bottle): MVTec photographs these parts in a fixed pose,
  so a defect keeps its position relative to the part's centre, plus a small
  jitter. "orbit" optionally rotates it about the part's centre, which is
  valid for round parts (bottle). This keeps the surrounding structure (lobes,
  rims) lined up; free placement pasted lobes at the wrong angle, which the
  first real metal_nut batch showed.
- "axis" (screw): parts lie at random angles. Both parts' centre and main axis
  are estimated from their foreground, and the defect is carried from one
  frame to the other, so a thread defect stays on the thread and a head
  scratch on the head. The 180-degree ambiguity of the axis is settled by
  whichever choice overlaps the host's silhouette better.
- "free" (texture categories): anywhere on the foreground, any rotation, flip, scale.

With ``register: true`` (aligned without orbit, and axis), the centroid/axis
estimate is then refined by image registration (OpenCV ECC, rotation plus
shift, the defect itself masked out), so the source part's edges land exactly
on the host's. Without it the faded patch around a long scratch left faint
double edges on the nut's lobes in the first real batch. Jitter then defaults
to zero for these parts, because any shift moves the structure out of line.

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
    mode: str = "free"  # "free" | "aligned" | "axis"
    rotate: str = "free"  # free mode: "free" (any angle), "small" (+-15 deg), "none"
    flip: bool = True  # free mode only
    scale: tuple[float, float] = (0.8, 1.25)  # free mode only; aligned/axis keep the size
    orbit: str = "none"  # aligned mode: "none", "any" (round parts) or "90" (four-fold parts)
    jitter: float = 0.02  # aligned/axis: shift as a fraction of the image size
    register: bool = False  # aligned (orbit "none") and axis: refine the pose by ECC image registration
    min_alignment: float = 0.9
    max_tries: int = 50
    foreground: str = "auto"  # "auto" or "full" (texture categories)

    def __post_init__(self):
        if self.mode not in ("free", "aligned", "axis"):
            raise ValueError(f"Unknown placement mode {self.mode!r}")
        if self.orbit not in ("none", "any", "90"):
            raise ValueError(f"Unknown orbit {self.orbit!r}")
        self.scale = tuple(self.scale)


@dataclass
class PlacedMask:
    mask: np.ndarray  # bool, host image size
    source_index: int  # index into MaskBank.sources
    affine: np.ndarray  # 2x3, source pixel coords -> host pixel coords
    alignment: float
    defect_type: str


@dataclass
class Pose:
    cx: float
    cy: float
    angle: float  # main-axis angle in radians, image coordinates (y down)


def part_pose(fg: np.ndarray) -> Pose:
    m = cv2.moments(fg.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        h, w = fg.shape
        return Pose(w / 2, h / 2, 0.0)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    angle = 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])
    return Pose(cx, cy, angle)


def _rot(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def _frame_affine(src: Pose, dst_xy: tuple[float, float], theta: float) -> np.ndarray:
    """p' = R(theta) (p - c_src) + dst."""
    lin = _rot(theta)
    t = np.array(dst_xy) - lin @ np.array([src.cx, src.cy])
    return np.hstack([lin, t[:, None]]).astype(np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


class MaskBank:
    def __init__(self, defects: list[Sample], policy: PlacementPolicy | None = None):
        if not defects:
            raise ValueError("MaskBank needs at least one training defect")
        self.policy = policy or PlacementPolicy()
        self.sources = list(defects)
        self._masks: dict[int, np.ndarray] = {}
        self._images: dict[int, np.ndarray] = {}
        self._fg: dict[int, np.ndarray] = {}
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

    def source_foreground(self, i: int) -> np.ndarray:
        """The source part's silhouette, with the defect itself counted as part (a broken edge is still the part)."""
        if i not in self._fg:
            self._fg[i] = estimate_foreground(self.source_image(i), self.policy.foreground) | self.source_mask(i)
        return self._fg[i]

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
        self._host = host
        best: PlacedMask | None = None
        for _ in range(p.max_tries):
            idx = int(rng.choice(self.by_type[defect_type]))
            src_mask = self.source_mask(idx)
            if not src_mask.any():
                continue
            if p.mode == "free":
                affine = self._free_affine(src_mask, fg, rng, (h, w))
            else:
                affine = self._part_affine(idx, fg, rng, (h, w))
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

    def _free_affine(self, src_mask: np.ndarray, fg: np.ndarray, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        p = self.policy
        ys, xs = np.nonzero(src_mask)
        cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        angle = {"free": rng.uniform(0, 360), "small": rng.uniform(-15, 15), "none": 0.0}[p.rotate]
        scale = rng.uniform(*p.scale)
        fx = -1.0 if (p.flip and rng.random() < 0.5) else 1.0
        fy = -1.0 if (p.flip and rng.random() < 0.5) else 1.0
        pts = np.argwhere(fg)
        if len(pts) == 0:
            pts = np.argwhere(np.ones(shape, bool))
        ty, tx = pts[rng.integers(len(pts))]
        return _affine(cx, cy, float(tx), float(ty), angle, scale, fx, fy)

    def _part_affine(self, idx: int, host_fg: np.ndarray, rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        p = self.policy
        h, w = shape
        src = part_pose(self.source_foreground(idx))
        dst = part_pose(host_fg)
        if p.mode == "aligned":
            theta = {"none": 0.0, "any": rng.uniform(0, 2 * math.pi), "90": rng.integers(4) * math.pi / 2}[p.orbit]
        else:  # axis: turn the source's main axis onto the host's; pick the 180-degree branch that overlaps best
            base = dst.angle - src.angle
            sfg = self.source_foreground(idx).astype(np.uint8)
            scores = []
            for theta in (base, base + math.pi):
                a = _frame_affine(src, (dst.cx, dst.cy), theta)
                scores.append(_iou(cv2.warpAffine(sfg, a, (w, h), flags=cv2.INTER_NEAREST) > 0, host_fg))
            theta = base if scores[0] >= scores[1] else base + math.pi
        j = p.jitter * max(h, w)
        affine = _frame_affine(src, (dst.cx + rng.uniform(-j, j), dst.cy + rng.uniform(-j, j)), theta)
        if p.register and (p.mode == "axis" or p.orbit == "none"):
            affine = self._register(idx, host_fg, affine) if self._host is not None else affine
        return affine

    _host: np.ndarray | None = None

    def _register(self, idx: int, host_fg: np.ndarray, init: np.ndarray, size: int = 320) -> np.ndarray:
        """Refine a source->host rigid transform with ECC on downscaled grey images; keep ``init`` if ECC fails."""
        host = self._host
        src = self.source_image(idx)
        h, w = host.shape[:2]
        s = size / max(h, w, *src.shape[:2])
        def grey(img):
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            return cv2.resize(g, (max(8, round(img.shape[1] * s)), max(8, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        tmpl, inp = grey(host), grey(src)
        # Valid source pixels: the part, minus the defect (it has no counterpart on the host).
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        valid = self.source_foreground(idx) & ~(cv2.dilate(self.source_mask(idx).astype(np.uint8), k) > 0)
        valid = cv2.resize(valid.astype(np.uint8), inp.shape[::-1], interpolation=cv2.INTER_NEAREST)
        # ECC wants the host->source warp in downscaled pixels.
        inv = cv2.invertAffineTransform(init).astype(np.float32)
        inv[:, 2] *= s
        try:
            cc, warp = cv2.findTransformECC(
                tmpl, inp, inv, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), valid, 5
            )
        except cv2.error:
            return init
        if cc < 0.5:
            return init
        warp = warp.astype(np.float64)
        warp[:, 2] /= s
        return cv2.invertAffineTransform(warp)


def _affine(cx: float, cy: float, tx: float, ty: float, angle_deg: float, scale: float, fx: float, fy: float) -> np.ndarray:
    """2x3 matrix: move (cx, cy) to the origin, flip, rotate and scale, then move to (tx, ty)."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * scale
    lin = rot @ np.diag([fx, fy])
    t = np.array([tx, ty]) - lin @ np.array([cx, cy])
    return np.hstack([lin, t[:, None]]).astype(np.float64)
