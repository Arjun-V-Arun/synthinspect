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


def _part(shape, angle_deg, center, head=True):
    """An elongated part (a screw seen from above) with a wider head at one end."""
    img = np.full(shape + (3,), 30, np.uint8)
    rect = (center, (160, 36), angle_deg)
    import cv2

    cv2.fillPoly(img, [cv2.boxPoints(rect).astype(np.int32)], (200, 200, 200))
    a = np.deg2rad(angle_deg)
    hx, hy = int(center[0] + 70 * np.cos(a)), int(center[1] + 70 * np.sin(a))
    if head:
        cv2.circle(img, (hx, hy), 30, (200, 200, 200), -1)
    return img, (hx, hy)


def _write_defect(tmp_path, img, mask, name):
    from synthinspect.data.mvtec import Sample
    from synthinspect.io import write_mask, write_rgb

    write_rgb(tmp_path / f"{name}.png", img)
    write_mask(tmp_path / f"{name}_mask.png", mask)
    return Sample(str(tmp_path / f"{name}.png"), "head", str(tmp_path / f"{name}_mask.png"))


def test_axis_mode_carries_a_head_defect_to_the_host_head(tmp_path):
    src, (hx, hy) = _part((256, 256), 0, (128, 128))
    mask = np.zeros((256, 256), bool)
    mask[hy - 6 : hy + 6, hx - 6 : hx + 6] = True  # a defect on the head
    sample = _write_defect(tmp_path, src, mask, "src")
    for angle, center in [(60, (120, 136)), (-130, (140, 120)), (180, (128, 128))]:
        host, (thx, thy) = _part((256, 256), angle, center)
        bank = MaskBank([sample], PlacementPolicy(mode="axis", jitter=0.0))
        p = bank.sample(host, "head", np.random.default_rng(0))
        ys, xs = np.nonzero(p.mask)
        assert abs(xs.mean() - thx) < 8 and abs(ys.mean() - thy) < 8, (angle, xs.mean(), ys.mean(), thx, thy)


def test_aligned_mode_keeps_the_position_on_the_part(tmp_path):
    src, _ = _part((256, 256), 0, (128, 128), head=False)
    mask = np.zeros((256, 256), bool)
    mask[120:130, 180:190] = True  # right end of the part
    sample = _write_defect(tmp_path, src, mask, "src")
    host, _ = _part((256, 256), 0, (110, 140), head=False)  # same pose, shifted
    bank = MaskBank([sample], PlacementPolicy(mode="aligned", jitter=0.0))
    p = bank.sample(host, "head", np.random.default_rng(0))
    ys, xs = np.nonzero(p.mask)
    assert abs(xs.mean() - (184.5 - 18)) < 3 and abs(ys.mean() - (124.5 + 12)) < 3
    # orbit "any" moves it around the part's centre but keeps its distance from it
    bank = MaskBank([sample], PlacementPolicy(mode="aligned", orbit="any", jitter=0.0, min_alignment=0.0))
    for seed in range(5):
        p = bank.sample(host, "head", np.random.default_rng(seed))
        ys, xs = np.nonzero(p.mask)
        assert abs(np.hypot(xs.mean() - 110, ys.mean() - 140) - np.hypot(184.5 - 128, 124.5 - 128)) < 4


def test_bad_policy_values_are_rejected():
    with pytest.raises(ValueError):
        PlacementPolicy(mode="random")
    with pytest.raises(ValueError):
        PlacementPolicy(orbit="45")


def _textured_part(size=256, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 25, np.uint8)
    cv2.ellipse(img, (128, 128), (90, 60), 0, 0, 360, (180, 180, 180), -1)
    for _ in range(25):  # surface features for registration to lock onto
        c = (int(rng.integers(60, 196)), int(rng.integers(85, 171)))
        cv2.circle(img, c, int(rng.integers(3, 9)), tuple(int(v) for v in rng.integers(60, 250, 3)), -1)
    cv2.rectangle(img, (190, 118), (215, 138), (40, 40, 40), -1)  # a notch: makes the part asymmetric
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_registration_lines_up_a_slightly_rotated_part(tmp_path):
    import cv2

    src = _textured_part()
    mask = np.zeros((256, 256), bool)
    mask[95:103, 150:170] = True  # defect up and to the right of centre
    sample = _write_defect(tmp_path, src, mask, "src")
    # Host: the same part turned 6 degrees and shifted, like two nuts shot in "the same" pose
    true = cv2.getRotationMatrix2D((128, 128), 6, 1.0)
    true[:, 2] += (10, -7)
    host = cv2.warpAffine(src, true, (256, 256), borderValue=(25, 25, 25))
    expect = cv2.warpAffine(mask.astype(np.uint8), true, (256, 256), flags=cv2.INTER_NEAREST) > 0
    ey, ex = np.nonzero(expect)

    def err(policy):
        p = MaskBank([sample], policy).sample(host, "head", np.random.default_rng(0))
        ys, xs = np.nonzero(p.mask)
        return float(np.hypot(xs.mean() - ex.mean(), ys.mean() - ey.mean()))

    rough = err(PlacementPolicy(mode="aligned", jitter=0.0, min_alignment=0.0))
    fine = err(PlacementPolicy(mode="aligned", jitter=0.0, register=True, min_alignment=0.0))
    assert fine < 1.5 and fine < rough / 2, (fine, rough)


def test_dataset_leaves_out_generated_images_of_skipped_types(split, tmp_path):
    """Images generated before a type was listed under skip_synthesis must not reach training."""
    import yaml

    from synthinspect.data.yolo_format import write_condition
    from synthinspect.gen.base import generate
    from synthinspect.gen.copypaste import CopyPasteGenerator

    generate(CopyPasteGenerator(), split, MaskBank(split.defect_train), tmp_path / "cp", n_per_type=3, seed=0, progress_every=None)
    out = tmp_path / "ds"
    write_condition(split, out, [tmp_path / "cp"], n_generated_per_type=3, skip_types=["stain"])
    counts = yaml.safe_load((out / "counts.yaml").read_text())
    assert counts["synthetic"] == 3 and counts["synthetic_skipped"] == 3
    names = [p.name for p in (out / "train" / "images").iterdir()]
    assert not any(n.startswith("syn_") and "stain" in n for n in names)
    assert any(n.startswith("real_stain") for n in names)  # the real stains still train
