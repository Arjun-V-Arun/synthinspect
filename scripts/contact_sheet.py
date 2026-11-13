"""Tile a few generated images per defect type into one picture, each defect outlined in red.

    python scripts/contact_sheet.py runs/bottle/gen/copypaste
writes runs/bottle/gen/copypaste/sheet.png: one row per defect type, the first
``--per-type`` images of each. ``--types`` keeps only some defect types, and
``--cols`` wraps long rows (each type still starts on a new row):
    python scripts/contact_sheet.py runs/screw/gen/copypaste --types manipulated_front --per-type 24 --cols 6

The outline sits just outside the defect mask, so it shows where the defect was
placed without hiding it.

Needs only numpy and OpenCV, so it runs without installing the package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def tile(image: np.ndarray, mask: np.ndarray, size: int, label: str) -> np.ndarray:
    ring = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = image.copy()
    cv2.drawContours(img, contours, -1, (0, 0, 255), max(1, round(max(img.shape[:2]) / size)))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (size, 18), (0, 0, 0), -1)
    cv2.putText(img, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path, help="a generator output folder (has images/, masks/, meta.jsonl)")
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--size", type=int, default=320, help="tile side in pixels")
    ap.add_argument("--out", type=Path, help="default: <folder>/sheet.png")
    ap.add_argument("--types", nargs="*", help="defect types to include (default: all)")
    ap.add_argument("--cols", type=int, help="tiles per row (default: --per-type)")
    a = ap.parse_args(argv)

    meta = a.folder / "meta.jsonl"
    if meta.exists():
        rows = [json.loads(ln) for ln in meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
        by_type: dict[str, list[str]] = {}
        for r in rows:
            by_type.setdefault(r["defect_type"], []).append(r["file"])
    else:
        by_type = {}
        for p in sorted((a.folder / "images").glob("*.png")):
            by_type.setdefault(p.stem.rsplit("_", 1)[0], []).append(p.name)
    if a.types:
        unknown = set(a.types) - set(by_type)
        if unknown:
            raise SystemExit(f"Not in {a.folder}: {sorted(unknown)}; it has {sorted(by_type)}")
        by_type = {t: by_type[t] for t in a.types}
    if not by_type:
        raise SystemExit(f"No generated images found in {a.folder}")
    cols = a.cols or a.per_type

    blank = np.zeros((a.size, a.size, 3), np.uint8)
    grid = []
    for dtype in sorted(by_type):
        tiles = []
        for name in sorted(by_type[dtype])[: a.per_type]:
            image = cv2.imread(str(a.folder / "images" / name), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(a.folder / "masks" / name), cv2.IMREAD_GRAYSCALE)
            tiles.append(tile(image, mask > 0, a.size, name[:-4]) if image is not None and mask is not None else blank)
        tiles += [blank] * (-(-a.per_type // cols) * cols - len(tiles))  # pad to whole rows
        grid += [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    out = a.out or a.folder / "sheet.png"
    cv2.imwrite(str(out), np.vstack(grid))
    print(f"wrote {out} ({len(by_type)} defect types x {a.per_type})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
