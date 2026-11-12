"""Shared generation loop: pick a good part, place a mask, let the generator paint it, save it."""

from __future__ import annotations

import time
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from ..data.foreground import estimate_foreground
from ..data.maskbank import MaskBank, PlacedMask
from ..data.mvtec import Split
from ..io import append_jsonl, read_jsonl, read_rgb, write_mask, write_rgb


class Generator(Protocol):
    name: str

    def fit(self, split: Split, bank: MaskBank, workdir: Path) -> None:
        """Learn whatever the method needs from the training defects only."""

    def paint(self, host: np.ndarray, placed: PlacedMask, bank: MaskBank, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (image, label_mask, info) for one synthetic defect on ``host``."""


def generate(
    gen: Generator,
    split: Split,
    bank: MaskBank,
    out_dir: str | Path,
    n_per_type: int,
    seed: int = 0,
    foreground_mode: str = "auto",
    resume: bool = True,
) -> list[dict]:
    """Write ``n_per_type`` images per defect type to ``out_dir``; resumable."""
    out_dir = Path(out_dir)
    meta_path = out_dir / "meta.jsonl"
    done = {r["file"] for r in read_jsonl(meta_path)} if resume else set()
    hosts = [s.image for s in split.normal_train]
    fg_cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for dtype in split.defect_types:
        if dtype not in bank.by_type:
            continue
        for i in range(n_per_type):
            name = f"{dtype}_{i:05d}.png"
            if name in done:
                continue
            # A per-image seed makes any single image reproducible and makes resume deterministic.
            # zlib.crc32 is stable across processes, unlike hash() on strings.
            rng = np.random.default_rng([seed, zlib.crc32(dtype.encode("utf-8")), i])
            host_path = hosts[int(rng.integers(len(hosts)))]
            host = read_rgb(host_path)
            if host_path not in fg_cache:
                fg_cache[host_path] = estimate_foreground(host, foreground_mode)
            placed = bank.sample(host, dtype, rng, host_fg=fg_cache[host_path])
            t0 = time.perf_counter()
            img, label, info = gen.paint(host, placed, bank, rng)
            if img.shape != host.shape:
                raise ValueError(f"{gen.name} returned shape {img.shape}, expected {host.shape}")
            write_rgb(out_dir / "images" / name, img)
            write_mask(out_dir / "masks" / name, label)
            row = {
                "file": name,
                "defect_type": dtype,
                "host": host_path,
                "source": bank.sources[placed.source_index].image,
                "alignment": round(placed.alignment, 4),
                "seconds": round(time.perf_counter() - t0, 3),
                "generator": gen.name,
                **info,
            }
            append_jsonl(meta_path, [row])
            rows.append(row)
    return rows
