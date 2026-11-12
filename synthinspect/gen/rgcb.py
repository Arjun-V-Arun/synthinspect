"""The RGCB approach: labelled composite, then neural style transfer toward a real image.

At RGCB the pipeline made labelled synthetic plates and used NST so they looked
like real ones while keeping colony structure (and so the labels). Here the
labelled composite is a copy-pasted real defect on a new good part, and the
style image is another real image of the same part.

Knobs that should match what was actually done at RGCB (edit configs/generators.yaml):
- ``scope``: "global" restyles the whole composite (closest to restyling whole
  synthetic plates); "local" keeps the restyled pixels only around the defect.
- ``style_source``: "normal" (a good part) or "defect" (another training defect).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..data.maskbank import MaskBank, PlacedMask
from ..data.mvtec import Split
from ..io import read_rgb
from .copypaste import CopyPasteGenerator, feather_alpha
from .nst import StyleTransfer


class RGCBGenerator:
    name = "rgcb"

    def __init__(
        self,
        scope: str = "global",
        style_source: str = "normal",
        local_margin: int = 24,
        blend: str = "poisson",
        nst: dict | None = None,
        pretrained_vgg: bool = True,
    ):
        if scope not in ("global", "local"):
            raise ValueError("scope must be 'global' or 'local'")
        if style_source not in ("normal", "defect"):
            raise ValueError("style_source must be 'normal' or 'defect'")
        self.scope, self.style_source, self.local_margin = scope, style_source, local_margin
        self.cp = CopyPasteGenerator(blend=blend)
        self.nst_cfg = dict(nst or {})
        self.pretrained_vgg = pretrained_vgg
        self._nst: StyleTransfer | None = None
        self._styles: list[str] = []
        self._style_cache: dict[str, np.ndarray] = {}

    def fit(self, split: Split, bank: MaskBank, workdir: Path) -> None:
        pool = split.normal_train if self.style_source == "normal" else split.defect_train
        self._styles = [s.image for s in pool]
        if not self._styles:
            raise ValueError("No style images available")

    @property
    def nst(self) -> StyleTransfer:
        if self._nst is None:
            self._nst = StyleTransfer(pretrained=self.pretrained_vgg, **self.nst_cfg)
        return self._nst

    def _style(self, path: str) -> np.ndarray:
        if path not in self._style_cache:
            if len(self._style_cache) > 64:
                self._style_cache.clear()
            self._style_cache[path] = read_rgb(path)
        return self._style_cache[path]

    def paint(self, host: np.ndarray, placed: PlacedMask, bank: MaskBank, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        composite, label, info = self.cp.paint(host, placed, bank, rng)
        style_path = self._styles[int(rng.integers(len(self._styles)))]
        styled = self.nst(composite, self._style(style_path))
        if self.scope == "local":
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.local_margin + 1, 2 * self.local_margin + 1))
            region = cv2.dilate(label.astype(np.uint8), k) > 0
            a = feather_alpha(region, self.local_margin // 2)[..., None]
            styled = (a * styled + (1 - a) * composite).round().clip(0, 255).astype(np.uint8)
        info = {**info, "style": style_path, "scope": self.scope}
        return styled, label, info
