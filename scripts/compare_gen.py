"""Compare generated image sets, or summarise one.

Several folders: each is compared with the FIRST, image by image.
    python scripts/compare_gen.py runs/metal_nut/gen/copypaste runs/metal_nut/gen/rgcb_fp32 runs/metal_nut/gen/rgcb
For each shared file it prints the mean absolute pixel difference (0-255) over the
whole image, inside the defect mask and outside it, and the seconds each took.
With copy-paste first, that is exactly what the style transfer changed, since RGCB
starts from the same composite.

One folder: speed, and (for RGCB) how much the output still changed late in the
style-transfer optimisation, from the trace each image records.
    python scripts/compare_gen.py runs/metal_nut/gen/rgcb

Needs only numpy and OpenCV, so it runs without installing the package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_meta(folder: Path) -> dict[str, dict]:
    path = folder / "meta.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: is this a generator output folder?")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["file"]: r for r in rows}


def summarise(folder: Path, meta: dict[str, dict]) -> None:
    secs = [r["seconds"] for r in meta.values() if "seconds" in r]
    print(f"{folder}: {len(meta)} images, {np.mean(secs):.1f} s/image on average" if secs else f"{folder}: {len(meta)} images")
    traces = [r["nst_trace"] for r in meta.values() if "nst_trace" in r]
    warm = [t for t in traces if not t.get("cold")]
    if not traces:
        return
    use = warm or traces
    label = "warm-started images" if warm else "cold-started images only (no warm ones yet)"
    print(f"  style transfer, {label} ({len(use)}): mean change (0-255) from the final image had it stopped after")
    for k in sorted({k for t in use for k in t.get("change_after", {})}, key=int):
        vals = [t["change_after"][k] for t in use if k in t.get("change_after", {})]
        print(f"    {int(k):>4} steps: {np.mean(vals):.2f} (max {np.max(vals):.2f})")
    amp = {t.get("amp") for t in traces}
    print(f"  bf16 (amp): {', '.join(str(a) for a in sorted(amp, key=str))}")


def compare(a: Path, b: Path, top: int | None = None) -> None:
    ma, mb = load_meta(a), load_meta(b)
    shared = sorted(set(ma) & set(mb))
    print(f"{b} compared with {a} ({len(shared)} shared files):")
    if not shared:
        print("  no file names in common")
        return
    whole, inside, outside, lines = [], [], [], []
    for name in shared:
        ia = cv2.imread(str(a / "images" / name), cv2.IMREAD_COLOR)
        ib = cv2.imread(str(b / "images" / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(a / "masks" / name), cv2.IMREAD_GRAYSCALE) > 0
        if ia is None or ib is None or ia.shape != ib.shape:
            print(f"  {name:<22} missing or different size, skipped")
            continue
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16)).mean(axis=2)
        w, i, o = d.mean(), d[mask].mean() if mask.any() else float("nan"), d[~mask].mean()
        whole.append(w), inside.append(i), outside.append(o)
        sa, sb = ma[name].get("seconds", float("nan")), mb[name].get("seconds", float("nan"))
        lines.append((w, f"  {name:<22}{w:>8.2f}{i:>8.2f}{o:>9.2f}{sa:>8.1f}{sb:>8.1f}"))
    if top is not None:
        lines = sorted(lines, key=lambda t: -t[0])[:top]
        print(f"  the {len(lines)} most changed:")
    print(f"  {'file':<22}{'whole':>8}{'defect':>8}{'outside':>9}{'sec A':>8}{'sec B':>8}")
    for _, line in lines:
        print(line)
    if whole:
        print(f"  {'mean':<22}{np.mean(whole):>8.2f}{np.nanmean(inside):>8.2f}{np.mean(outside):>9.2f}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="+", type=Path, help="one folder to summarise, or several to compare with the first")
    ap.add_argument("--top", type=int, help="list only the N most changed files (the mean still covers all)")
    a = ap.parse_args(argv)
    for other in a.folders[1:]:
        compare(a.folders[0], other, a.top)
    for folder in a.folders:
        summarise(folder, load_meta(folder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
