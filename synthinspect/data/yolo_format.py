"""Write a training condition as a YOLO-format dataset (read by both Ultralytics and RF-DETR).

A condition = the few real training defects + good parts as negatives
(+ synthetic defects from one generator). There is deliberately no real
validation set: the only labelled real images are the training defects, and
the test set is off limits. "valid" is a copy of the real training images so
both libraries run, but no checkpoint is chosen with it: training runs a fixed
number of epochs and the last checkpoint is used.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from ..io import read_jsonl, read_mask
from .mvtec import Split, boxes_from_mask


def yolo_lines(mask: np.ndarray, cls: int, min_area: int, merge_frac: float) -> list[str]:
    h, w = mask.shape
    boxes = boxes_from_mask(mask, min_area=min_area, merge_dist=max(1, int(merge_frac * max(h, w))))
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _put(img_src: str | Path, lines: list[str], images_dir: Path, labels_dir: Path, stem: str) -> None:
    suffix = Path(img_src).suffix.lower() or ".png"
    shutil.copyfile(img_src, images_dir / f"{stem}{suffix}")
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_condition(
    split: Split,
    out_dir: str | Path,
    generated_dirs: list[str | Path] | None = None,
    n_generated_per_type: int | None = None,
    n_normals: int | None = None,
    class_mode: str = "binary",
    include_real: bool = True,
    min_area: int = 9,
    merge_frac: float = 0.01,
    seed: int = 0,
) -> Path:
    """Build the dataset folder and return the path of its data.yaml."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    names = ["defect"] if class_mode == "binary" else list(split.defect_types)
    cls_of = (lambda t: 0) if class_mode == "binary" else (lambda t: names.index(t))
    rng = np.random.default_rng(seed)

    tr_i, tr_l = out / "train" / "images", out / "train" / "labels"
    va_i, va_l = out / "valid" / "images", out / "valid" / "labels"
    for d in (tr_i, tr_l, va_i, va_l):
        d.mkdir(parents=True)

    counts = {"real_defect": 0, "normal": 0, "synthetic": 0}
    if include_real:
        for i, s in enumerate(split.defect_train):
            lines = yolo_lines(read_mask(s.mask), cls_of(s.defect_type), min_area, merge_frac)
            _put(s.image, lines, tr_i, tr_l, f"real_{s.defect_type}_{i:03d}")
            _put(s.image, lines, va_i, va_l, f"real_{s.defect_type}_{i:03d}")
            counts["real_defect"] += 1

    normals = list(split.normal_train)
    if n_normals is not None and n_normals < len(normals):
        normals = [normals[j] for j in sorted(rng.choice(len(normals), n_normals, replace=False))]
    for i, s in enumerate(normals):
        _put(s.image, [], tr_i, tr_l, f"good_{i:04d}")
        counts["normal"] += 1
    for i, s in enumerate(normals[:10]):  # a few negatives so "valid" is not all defects
        _put(s.image, [], va_i, va_l, f"good_{i:04d}")

    for gdir in generated_dirs or []:
        gdir = Path(gdir)
        rows = read_jsonl(gdir / "meta.jsonl")
        per_type: dict[str, int] = {}
        for r in sorted(rows, key=lambda r: r["file"]):
            t = r["defect_type"]
            if t not in split.defect_types:
                continue
            if n_generated_per_type is not None and per_type.get(t, 0) >= n_generated_per_type:
                continue
            lines = yolo_lines(read_mask(gdir / "masks" / r["file"]), cls_of(t), min_area, merge_frac)
            if not lines:
                continue
            _put(gdir / "images" / r["file"], lines, tr_i, tr_l, f"syn_{gdir.name}_{Path(r['file']).stem}")
            per_type[t] = per_type.get(t, 0) + 1
            counts["synthetic"] += 1

    data = {"path": str(out.resolve()).replace("\\", "/"), "train": "train/images", "val": "valid/images", "names": dict(enumerate(names))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (out / "counts.yaml").write_text(yaml.safe_dump(counts), encoding="utf-8")
    return out / "data.yaml"
