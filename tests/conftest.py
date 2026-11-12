import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import make_mvtec  # noqa: E402


@pytest.fixture(scope="session")
def mvtec_root(tmp_path_factory):
    return make_mvtec(tmp_path_factory.mktemp("mvtec"))


@pytest.fixture(scope="session")
def split(mvtec_root):
    from synthinspect.data.mvtec import make_split

    return make_split(mvtec_root, "widget")


@pytest.fixture(scope="session")
def trained_yolo(split, tmp_path_factory):
    """A YOLO26n trained from scratch for a few epochs on the fixture (CPU, about 10 s)."""
    from synthinspect.data.yolo_format import write_condition
    from synthinspect.detect.yolo import train_yolo

    work = tmp_path_factory.mktemp("yolo")
    data = write_condition(split, work / "ds", n_normals=12)
    return train_yolo(data, work / "runs" / "scratch", pretrained=False, epochs=3, imgsz=128, batch=8, workers=0, device="cpu")
