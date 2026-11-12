"""MVTec AD scanning and the leakage-free few-shot split.

Split rule (AnomalyDiffusion / MAGIC convention): for each defect type, sort the
file names and take the first floor(n/3) as the few real training defects. The
rest, plus every test/good image, form the test set. Nothing in the test set is
ever used for training, generation, mask sampling, calibration of INT8 models or
checkpoint selection.

MVTec layout:
    <root>/<category>/train/good/*.png
    <root>/<category>/test/<type>/*.png          (type == "good" has no mask)
    <root>/<category>/ground_truth/<type>/<stem>_mask.png
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..io import read_json, read_mask, write_json

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image: str
    defect_type: str  # "good" for defect-free
    mask: str | None = None

    @property
    def is_defect(self) -> bool:
        return self.defect_type != "good"


@dataclass
class Split:
    category: str
    root: str
    normal_train: list[Sample] = field(default_factory=list)  # train/good, used as backgrounds and negatives
    defect_train: list[Sample] = field(default_factory=list)  # first third of each defect type
    test: list[Sample] = field(default_factory=list)  # remaining defects + all test/good
    defect_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "root": self.root,
            "defect_types": self.defect_types,
            "normal_train": [asdict(s) for s in self.normal_train],
            "defect_train": [asdict(s) for s in self.defect_train],
            "test": [asdict(s) for s in self.test],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Split":
        return cls(
            category=d["category"],
            root=d["root"],
            defect_types=list(d["defect_types"]),
            normal_train=[Sample(**s) for s in d["normal_train"]],
            defect_train=[Sample(**s) for s in d["defect_train"]],
            test=[Sample(**s) for s in d["test"]],
        )

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        return cls.from_dict(read_json(path))

    def check_disjoint(self) -> None:
        """Raise if any image appears in both the training and the test side."""
        train = {s.image for s in self.normal_train} | {s.image for s in self.defect_train}
        test = {s.image for s in self.test}
        overlap = train & test
        if overlap:
            raise AssertionError(f"Train/test overlap: {sorted(overlap)[:5]}")


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _mask_for(root: Path, category: str, defect_type: str, image: Path) -> Path:
    mask = root / category / "ground_truth" / defect_type / f"{image.stem}_mask.png"
    if not mask.exists():
        raise FileNotFoundError(f"Missing ground-truth mask {mask}")
    return mask


def make_split(
    root: str | Path,
    category: str,
    train_fraction_den: int = 3,
    exclude_types: list[str] | None = None,
) -> Split:
    """Build the few-shot split for one category.

    ``train_fraction_den=3`` means floor(n/3) training defects per type, never fewer than 1.
    """
    root = Path(root)
    cat = root / category
    if not cat.is_dir():
        raise FileNotFoundError(f"Category folder not found: {cat}")
    exclude = set(exclude_types or [])

    split = Split(category=category, root=str(root).replace("\\", "/"))
    split.normal_train = [Sample(str(p).replace("\\", "/"), "good") for p in _images(cat / "train" / "good")]
    if not split.normal_train:
        raise FileNotFoundError(f"No train/good images in {cat}")

    test_root = cat / "test"
    for type_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        dtype = type_dir.name
        files = _images(type_dir)
        if dtype == "good":
            split.test.extend(Sample(str(p).replace("\\", "/"), "good") for p in files)
            continue
        if dtype in exclude:
            continue
        samples = [Sample(str(p).replace("\\", "/"), dtype, str(_mask_for(root, category, dtype, p)).replace("\\", "/")) for p in files]
        k = max(1, len(samples) // train_fraction_den)
        split.defect_train.extend(samples[:k])
        split.test.extend(samples[k:])
        split.defect_types.append(dtype)

    split.check_disjoint()
    return split


def boxes_from_mask(mask: np.ndarray, min_area: int = 9, merge_dist: int = 0) -> np.ndarray:
    """Axis-aligned boxes (x1, y1, x2, y2), one per connected component.

    ``merge_dist`` dilates the mask first so fragments of one defect (a broken
    scratch) become one box. Components smaller than ``min_area`` pixels are dropped.
    """
    m = (mask > 0).astype(np.uint8)
    if merge_dist > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_dist + 1, 2 * merge_dist + 1))
        grouped = cv2.dilate(m, k)
    else:
        grouped = m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    boxes = []
    for i in range(1, n):
        comp = (labels == i) & (m > 0)
        if comp.sum() < min_area:
            continue
        ys, xs = np.nonzero(comp)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def gt_for_sample(sample: Sample, min_area: int = 9, merge_dist: int = 8) -> tuple[np.ndarray, np.ndarray | None]:
    """Ground-truth boxes and mask for a test sample (empty for good images)."""
    if not sample.is_defect:
        return np.zeros((0, 4), np.float32), None
    mask = read_mask(sample.mask)
    return boxes_from_mask(mask, min_area=min_area, merge_dist=merge_dist), mask
