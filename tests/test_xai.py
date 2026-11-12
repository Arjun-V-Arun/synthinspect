import numpy as np
import pytest

from synthinspect.detect.base import Detections
from synthinspect.xai.drise import drise, random_masks
from synthinspect.xai.metrics import box_region, deletion_insertion, energy_in, pointing_game, saliency_agreement


class SquareDetector:
    """Finds one bright square; its score is the square's visible brightness. A model whose 'reasoning' we know."""

    name, imgsz = "toy", 64
    box = np.array([20, 24, 36, 40], np.float32)

    def predict(self, images, min_score=0.0):
        out = []
        for im in images:
            x1, y1, x2, y2 = self.box.astype(int)
            score = float(im[y1:y2, x1:x2].mean() / 255.0)
            out.append(Detections(self.box[None], np.array([score], np.float32), np.zeros(1, np.int64)) if score >= min_score else Detections())
        return out


@pytest.fixture()
def scene():
    img = np.full((64, 64, 3), 20, np.uint8)
    img[24:40, 20:36] = 250
    return img


def test_random_masks_shape_and_density():
    m = random_masks(50, (40, 60), grid=8, p=0.5)
    assert m.shape == (50, 40, 60) and 0.35 < m.mean() < 0.65


def test_drise_finds_the_evidence(scene):
    det = SquareDetector()
    sal = drise(det, scene, det.box, n_masks=300, grid=8, batch=50)
    region = box_region(det.box, sal.shape)
    assert pointing_game(sal, region, tolerance=0)
    # D-RISE maps are smooth (masks are coarse grids), so the evidence is concentrated but not sharp:
    # the box gets well above its uniform share and is clearly hotter than the rest of the image.
    assert energy_in(sal, region) > 1.5 * region.mean()
    assert sal[region].mean() > 1.5 * sal[~region].mean()


def test_deletion_insertion_rank_good_map_above_random(scene):
    det = SquareDetector()
    good = box_region(det.box, scene.shape[:2]).astype(np.float32)
    rng = np.random.default_rng(0)
    bad = rng.random(scene.shape[:2]).astype(np.float32)
    g = deletion_insertion(det, scene, det.box, good, steps=8, block=4)
    b = deletion_insertion(det, scene, det.box, bad, steps=8, block=4)
    assert g["deletion"] < b["deletion"]
    assert g["insertion"] > b["insertion"]


def test_energy_and_pointing():
    sal = np.zeros((10, 10), np.float32)
    sal[2, 3] = 1.0
    sal[8, 8] = 0.5
    mask = np.zeros((10, 10), bool)
    mask[2, 3] = True
    assert pointing_game(sal, mask, tolerance=0)
    assert energy_in(sal, mask) == pytest.approx(1 / 1.5)


def test_agreement():
    rng = np.random.default_rng(0)
    a = rng.random((32, 32)).astype(np.float32)
    assert saliency_agreement(a, a, size=32)["spearman"] == pytest.approx(1.0)
    assert saliency_agreement(a, a, size=32)["top10_iou"] == pytest.approx(1.0)
    assert saliency_agreement(a, rng.random((32, 32)).astype(np.float32), size=32)["spearman"] < 0.3


def test_odam_on_yolo26(trained_yolo, split):
    from synthinspect.detect.yolo import TorchYOLO
    from synthinspect.io import read_rgb
    from synthinspect.xai.odam import odam

    det = TorchYOLO(trained_yolo, imgsz=128, device="cpu")
    img = read_rgb(next(s for s in split.test if s.is_defect).image)
    maps, raw = odam(det, img, indices=[0, 1])
    assert raw.shape[1] == 6 and len(maps) == 2
    for m in maps:
        assert m.shape == img.shape[:2] and m.max() == pytest.approx(1.0) and m.min() >= 0
    # Same detection explained twice gives the same map (no state leaks between backward passes).
    again, _ = odam(det, img, indices=[0])
    assert np.allclose(again[0], maps[0], atol=1e-5)
