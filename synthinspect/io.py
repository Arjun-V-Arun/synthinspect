"""Small I/O helpers that behave the same on Windows and Linux.

OpenCV's imread/imwrite fail silently on non-ASCII Windows paths, so images go
through numpy byte buffers instead. Images are always RGB uint8 inside the code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a mask as a boolean array (any non-zero pixel is foreground)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read mask: {path}")
    return img > 0


def write_rgb(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    buf.tofile(str(path))


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)
    if not ok:
        raise ValueError(f"Could not encode mask for {path}")
    buf.tofile(str(path))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    """Short hash of a JSON-serialisable object, used to name runs by their config."""
    blob = json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o).replace("\\", "/")
    raise TypeError(f"Not JSON serialisable: {type(o)}")
