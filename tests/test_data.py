import numpy as np
import pytest

from synthinspect.data.foreground import estimate_foreground
from synthinspect.data.maskbank import MaskBank, PlacementPolicy
from synthinspect.data.mvtec import Split, boxes_from_mask, gt_for_sample, make_split
from synthinspect.io import read_rgb


def test_split_takes_first_third_per_type_and_is_disjoint(split):
    by_type = {}
    for s in split.defect_train:
        by_type.setdefault(s.defect_type, []).append(s)
    assert set(by_type) == {"scratch", "stain"}
    assert all(len(v) == 3 for v in by_type.values())  # floor(9 / 3)
    # The training defects are the first files in sorted order.
    assert [s.image.split("/")[-1] for s in by_type["scratch"]] == ["000.png", "001.png", "002.png"]
    test_defects = [s for s in split.test if s.is_defect]
    assert len(test_defects) == 12 and sum(not s.is_defect for s in split.test) == 6
    split.check_disjoint()


def test_split_roundtrip(split, tmp_path):
    split.save(tmp_path / "split.json")
    again = Split.load(tmp_path / "split.json")
    assert again.to_dict() == split.to_dict()


def test_overlap_is_detected(split):
    bad = Split.from_dict(split.to_dict())
    bad.test.append(bad.defect_train[0])
    with pytest.raises(AssertionError):
        bad.check_disjoint()


def test_boxes_from_mask_merges_fragments():
    m = np.zeros((50, 50), bool)
    m[10:12, 5:15] = True
    m[10:12, 17:25] = True  # same scratch, 2-pixel gap
    m[40:45, 40:45] = True
    assert len(boxes_from_mask(m, min_area=4, merge_dist=0)) == 3
    boxes = boxes_from_mask(m, min_area=4, merge_dist=3)
    assert len(boxes) == 2
    assert boxes[0].tolist() == [5, 10, 25, 12]


def test_gt_for_good_sample_is_empty(split):
    good = next(s for s in split.test if not s.is_defect)
    boxes, mask = gt_for_sample(good)
    assert boxes.shape == (0, 4) and mask is None


def test_foreground_finds_the_part(split):
    img = read_rgb(split.normal_train[0].image)
    fg = estimate_foreground(img)
    h, w = fg.shape
    assert fg[h // 2 - 25, w // 2] and not fg[2, 2]
    assert 0.15 < fg.mean() < 0.5
    assert estimate_foreground(img, "full").all()


def test_maskbank_places_masks_on_the_part(split):
    bank = MaskBank(split.defect_train, PlacementPolicy(min_alignment=0.9))
    host = read_rgb(split.normal_train[3].image)
    fg = estimate_foreground(host)
    rng = np.random.default_rng(0)
    aligns = []
    for dtype in ("scratch", "stain"):
        for _ in range(10):
            p = bank.sample(host, dtype, rng, host_fg=fg)
            assert p.mask.shape == host.shape[:2] and p.mask.sum() > 0
            assert bank.sources[p.source_index].defect_type == dtype
            aligns.append(p.alignment)
    assert np.mean(np.array(aligns) >= 0.9) > 0.9


def test_maskbank_only_uses_training_defects(split):
    bank = MaskBank(split.defect_train)
    test_images = {s.image for s in split.test}
    assert not ({s.image for s in bank.sources} & test_images)


def test_affine_moves_mask_and_pixels_together(split):
    """Copy-paste relies on the mask and the pixels sharing one transform."""
    import cv2

    bank = MaskBank(split.defect_train)
    host = read_rgb(split.normal_train[0].image)
    p = bank.sample(host, "stain", np.random.default_rng(1))
    src_mask = bank.source_mask(p.source_index).astype(np.uint8)
    warped = cv2.warpAffine(src_mask, p.affine, host.shape[1::-1], flags=cv2.INTER_NEAREST) > 0
    assert (warped == p.mask).all()
