"""DefectFill and MAGIC as Generators: one LoRA adapter per defect type, trained once and cached."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data.maskbank import MaskBank, PlacedMask
from ..data.mvtec import Split
from ..io import read_mask, read_rgb, write_json
from . import sd_inpaint as sdi


class _SDGenerator:
    name = "sd"
    objective = ""

    def __init__(
        self,
        category: str,
        base_model: str = "sd-legacy/stable-diffusion-inpainting",
        object_name: str | None = None,
        train: dict | None = None,
        sample_steps: int = 30,
        resolution: int = 512,
        device: str | None = None,
        components: sdi.SDComponents | None = None,  # injected by tests
    ):
        self.category = category
        self.base_model = base_model
        self.object_name = object_name or category.replace("_", " ")
        self.train_cfg = dict(train or {})
        self.sample_steps = sample_steps
        self.resolution = resolution
        self.device = device
        self._comp = components
        self._loaded: set[str] = set()
        self.workdir: Path | None = None

    @property
    def comp(self) -> sdi.SDComponents:
        if self._comp is None:
            self._comp = sdi.load_components(self.base_model, self.device, self.resolution)
        return self._comp

    def _cfg(self, seed: int) -> sdi.TrainConfig:
        return sdi.TrainConfig(objective=self.objective, seed=seed, **self.train_cfg)

    def adapter_path(self, dtype: str) -> Path:
        assert self.workdir is not None
        return self.workdir / "lora" / f"{dtype}.pt"

    def fit(self, split: Split, bank: MaskBank, workdir: Path) -> None:
        self.workdir = Path(workdir)
        for dtype in split.defect_types:
            path = self.adapter_path(dtype)
            if path.exists():
                continue
            srcs = [s for s in split.defect_train if s.defect_type == dtype]
            images = [read_rgb(s.image) for s in srcs]
            masks = [read_mask(s.mask) for s in srcs]
            cfg = self._cfg(seed=len(dtype))
            state = sdi.train_lora(self.comp, images, masks, self.object_name, self._adapter_name(dtype), cfg)
            sdi.save_adapter(path, state)
            write_json(path.with_suffix(".json"), {"config": asdict(cfg), "n_images": len(images), "history": state["history"][-200:]})
            self._loaded.add(dtype)

    @staticmethod
    def _adapter_name(dtype: str) -> str:
        return "d_" + "".join(c if c.isalnum() else "_" for c in dtype)

    def _activate(self, dtype: str) -> None:
        name = self._adapter_name(dtype)
        cfg = self._cfg(seed=0)
        if dtype not in self._loaded:
            state = torch.load(self.adapter_path(dtype), map_location="cpu", weights_only=True)
            sdi.add_lora(self.comp.unet, cfg.rank, sdi.UNET_TARGETS, name)
            sdi.load_lora_state(self.comp.unet, state["unet"])
            if state.get("text"):
                sdi.add_lora(self.comp.text_encoder, cfg.rank, sdi.TEXT_TARGETS, name)
                sdi.load_lora_state(self.comp.text_encoder, state["text"])
            self._loaded.add(dtype)
        sdi.set_active(self.comp.unet, name)
        if self.objective == "defectfill" and cfg.train_text_encoder:
            sdi.set_active(self.comp.text_encoder, name)


class DefectFillGenerator(_SDGenerator):
    name = "defectfill"
    objective = "defectfill"

    def __init__(self, category: str, guidance: float = 7.5, lfs_candidates: int = 4, lpips_fn=None, **kw):
        super().__init__(category, **kw)
        self.guidance = guidance
        self.lfs_candidates = lfs_candidates
        self._lpips = lpips_fn

    def paint(self, host: np.ndarray, placed: PlacedMask, bank: MaskBank, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        self._activate(placed.defect_type)
        cands = sdi.inpaint(
            self.comp,
            host,
            placed.mask,
            f"A {self.object_name} with sks",
            n=self.lfs_candidates,
            steps=self.sample_steps,
            guidance=sdi.constant_guidance(self.guidance),
            seed=int(rng.integers(2**31)),
        )
        full = [sdi.composite(host, c, placed.mask) for c in cands]
        if self._lpips is None:
            self._lpips = sdi.load_lpips(self.comp.device)
        x1, y1, x2, y2 = sdi.crop_box(placed.mask)
        scores = [self._lpips(f[y1:y2, x1:x2], host[y1:y2, x1:x2]) for f in full]
        best = int(np.argmax(scores))  # Low-Fidelity Selection: the candidate least like the clean part
        return full[best], placed.mask.copy(), {"lfs_scores": [round(s, 4) for s in scores], "lfs_choice": best}


class MagicGenerator(_SDGenerator):
    name = "magic"
    objective = "magic"

    def __init__(self, category: str, w_d: float = 7.5, w_m: float = 4.0, **kw):
        super().__init__(category, **kw)
        self.w_d, self.w_m = w_d, w_m

    def paint(self, host: np.ndarray, placed: PlacedMask, bank: MaskBank, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        self._activate(placed.defect_type)
        cfg = self._cfg(seed=0)
        (img,) = sdi.inpaint(
            self.comp,
            host,
            placed.mask,
            sdi.PLACEHOLDER,
            n=1,
            steps=self.sample_steps,
            guidance=lambda i, n: sdi.magic_guidance(i, n, self.w_d, self.w_m),
            gpp_sigma=cfg.gpp_sigma,
            seed=int(rng.integers(2**31)),
        )
        return sdi.composite(host, img, placed.mask), placed.mask.copy(), {}
