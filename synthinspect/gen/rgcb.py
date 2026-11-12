"""The RGCB approach: real defects pasted onto clean parts, then the whole image restyled.

This mirrors the colony generator used at RGCB (Pawłowski, Majchrowska and
Golan, Scientific Reports 2022; github.com/jarek-pawlowski/microbial-dataset-generation):

1. Cut real objects (there: colonies; here: the few training defects) and paste
   them onto clean backgrounds (there: empty dishes; here: good parts) with a
   Gaussian fade. RGCB faded a square patch (sigma 0.8, ``blend: radial``);
   here the fade follows the defect's outline (``blend: alpha``), because the
   square patch's edge showed as straight seams on textured metal.
2. Restyle the WHOLE composite toward a real clean image with photorealistic
   HRNet style transfer (Li et al. 2019), 500 steps per image, warm-started
   from the network trained on the previous image.

What changes from colonies to defects: one defect per image rather than many
colonies, and placement comes from the shared mask bank so every generator
paints in the same places. Gatys pixel-optimisation NST is kept as an option
(``nst_method: gatys``) for comparison.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..data.maskbank import MaskBank, PlacedMask
from ..data.mvtec import Split
from ..io import read_rgb
from .copypaste import CopyPasteGenerator, feather_alpha


class RGCBGenerator:
    name = "rgcb"

    def __init__(
        self,
        scope: str = "global",
        style_source: str = "normal",
        local_margin: int = 24,
        blend: str = "alpha",
        feather: int = 12,
        context: int = 6,
        radial_sigma: float = 0.8,
        radial_pad: float = 0.25,
        nst_method: str = "hrnet",
        nst: dict | None = None,
        pretrained_vgg: bool = True,
    ):
        if scope not in ("global", "local"):
            raise ValueError("scope must be 'global' or 'local'")
        if style_source not in ("normal", "defect"):
            raise ValueError("style_source must be 'normal' or 'defect'")
        if nst_method not in ("hrnet", "gatys"):
            raise ValueError("nst_method must be 'hrnet' or 'gatys'")
        self.scope, self.style_source, self.local_margin = scope, style_source, local_margin
        self.cp = CopyPasteGenerator(blend=blend, feather=feather, context=context, radial_sigma=radial_sigma, radial_pad=radial_pad)
        self.nst_method = nst_method
        self.nst_cfg = dict(nst or {})
        self.pretrained_vgg = pretrained_vgg
        self._nst = None
        self._state_path: Path | None = None
        self._styles: list[str] = []
        self._style_cache: dict[str, np.ndarray] = {}

    def fit(self, split: Split, bank: MaskBank, workdir: Path) -> None:
        pool = split.normal_train if self.style_source == "normal" else split.defect_train
        self._styles = [s.image for s in pool]
        if not self._styles:
            raise ValueError("No style images available")
        # The warm-started HRNet is saved here after every image, so an interrupted run resumes with it.
        self._state_path = Path(workdir) / "nst_hrnet.pt"

    @property
    def nst(self):
        if self._nst is None:
            if self.nst_method == "hrnet":
                from .nst_hrnet import PhotorealisticNST

                self._nst = PhotorealisticNST(pretrained=self.pretrained_vgg, state_path=self._state_path, **self.nst_cfg)
            else:
                from .nst import StyleTransfer

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
        info = {**info, "style": style_path, "scope": self.scope, "nst": self.nst_method}
        return styled, label, info
