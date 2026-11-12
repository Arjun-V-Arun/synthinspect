import numpy as np
import pytest
import torch

from synthinspect.data.maskbank import MaskBank
from synthinspect.gen.base import generate
from synthinspect.gen.copypaste import CopyPasteGenerator
from synthinspect.gen.rgcb import RGCBGenerator
from synthinspect.io import read_jsonl, read_mask, read_rgb


@pytest.fixture()
def bank(split):
    return MaskBank(split.defect_train)


@pytest.mark.parametrize("blend", ["poisson", "alpha"])
def test_copypaste_changes_only_the_masked_region(split, bank, blend):
    gen = CopyPasteGenerator(blend=blend, context=2, feather=2)
    host = read_rgb(split.normal_train[0].image)
    rng = np.random.default_rng(0)
    placed = bank.sample(host, "stain", rng)
    img, label, info = gen.paint(host, placed, bank, rng)
    assert (label == placed.mask).all()
    far = ~_dilate(placed.mask, 12)
    assert np.abs(img.astype(int) - host.astype(int))[far].max() <= 2  # untouched away from the defect
    inside = np.abs(img.astype(int) - host.astype(int))[placed.mask].mean()
    assert inside > 10  # the stain actually arrived


def test_generate_writes_resumable_dataset(split, bank, tmp_path):
    gen = CopyPasteGenerator()
    rows = generate(gen, split, bank, tmp_path / "cp", n_per_type=3, seed=0)
    assert len(rows) == 6
    again = generate(gen, split, bank, tmp_path / "cp", n_per_type=3, seed=0)
    assert again == []  # nothing left to do
    meta = read_jsonl(tmp_path / "cp" / "meta.jsonl")
    assert {r["defect_type"] for r in meta} == {"scratch", "stain"}
    for r in meta:
        assert read_mask(tmp_path / "cp" / "masks" / r["file"]).any()
        assert r["source"] in {s.image for s in split.defect_train}


def test_generation_is_reproducible(split, bank, tmp_path):
    gen = CopyPasteGenerator()
    generate(gen, split, bank, tmp_path / "a", n_per_type=2, seed=7)
    generate(gen, split, bank, tmp_path / "b", n_per_type=2, seed=7)
    for f in ("scratch_00001.png", "stain_00000.png"):
        assert (read_rgb(tmp_path / "a" / "images" / f) == read_rgb(tmp_path / "b" / "images" / f)).all()


def test_rgcb_style_transfer_keeps_structure(split, bank, tmp_path):
    gen = RGCBGenerator(scope="local", local_margin=6, nst={"size": 64, "steps": 5, "device": "cpu"}, pretrained_vgg=False)
    gen.fit(split, bank, tmp_path)
    host = read_rgb(split.normal_train[1].image)
    rng = np.random.default_rng(3)
    placed = bank.sample(host, "scratch", rng)
    img, label, info = gen.paint(host, placed, bank, rng)
    assert img.shape == host.shape and (label == placed.mask).all()
    assert info["scope"] == "local"
    far = ~_dilate(placed.mask, 20)
    assert np.abs(img.astype(int) - host.astype(int))[far].max() <= 2  # local scope leaves the rest alone


def test_rgcb_global_scope_restyles_whole_image(split, bank, tmp_path):
    gen = RGCBGenerator(scope="global", nst={"size": 64, "steps": 3, "device": "cpu", "lr": 0.2}, pretrained_vgg=False)
    gen.fit(split, bank, tmp_path)
    host = read_rgb(split.normal_train[1].image)
    rng = np.random.default_rng(3)
    img, _, _ = gen.paint(host, bank.sample(host, "stain", rng), bank, rng)
    assert np.abs(img.astype(int) - host.astype(int)).mean() > 0.5


def _dilate(mask, r):
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0
