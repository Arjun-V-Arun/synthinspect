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


@pytest.mark.parametrize("blend", ["poisson", "alpha", "radial"])
def test_copypaste_changes_only_the_masked_region(split, bank, blend):
    gen = CopyPasteGenerator(blend=blend, context=2, feather=2, radial_pad=0.25)
    host = read_rgb(split.normal_train[0].image)
    rng = np.random.default_rng(0)
    placed = bank.sample(host, "stain", rng)
    img, label, info = gen.paint(host, placed, bank, rng)
    assert (label == placed.mask).all() and info["blend"] == blend
    if blend == "radial":  # the faded square patch may touch everything inside the padded box, nothing outside
        ys, xs = np.nonzero(_dilate(placed.mask, 2))
        pw, ph = int(round(0.25 * (xs.max() + 1 - xs.min()))), int(round(0.25 * (ys.max() + 1 - ys.min())))
        far = np.ones(host.shape[:2], bool)
        far[max(0, ys.min() - ph) : ys.max() + 1 + ph, max(0, xs.min() - pw) : xs.max() + 1 + pw] = False
    else:
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
    # alpha blend here so the only wide change could come from the style transfer, which "local" must confine
    gen = RGCBGenerator(scope="local", local_margin=6, blend="alpha", nst={"size": 64, "steps": 5, "device": "cpu"}, pretrained_vgg=False)
    gen.fit(split, bank, tmp_path)
    host = read_rgb(split.normal_train[1].image)
    rng = np.random.default_rng(3)
    placed = bank.sample(host, "scratch", rng)
    img, label, info = gen.paint(host, placed, bank, rng)
    assert img.shape == host.shape and (label == placed.mask).all()
    assert info["scope"] == "local"
    far = ~_dilate(placed.mask, 20)
    assert np.abs(img.astype(int) - host.astype(int))[far].max() <= 2  # local scope leaves the rest alone


def test_hrnet_nst_keeps_shape_and_warm_starts(tmp_path):
    import torch

    from synthinspect.gen.nst_hrnet import HRNet, PhotorealisticNST

    x = torch.rand(1, 3, 36, 52)
    assert HRNet()(x).shape == x.shape
    rng = np.random.default_rng(0)
    content = rng.integers(0, 255, (37, 50, 3), dtype=np.uint8)  # odd sizes: padding to a multiple of 4
    style = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    state = tmp_path / "nst_hrnet.pt"
    nst = PhotorealisticNST(size=48, steps=2, pretrained=False, state_path=state, device="cpu")
    out = nst(content, style)
    assert out.shape == content.shape and out.dtype == np.uint8
    assert state.exists()
    # A new instance (an interrupted run resuming) picks up the warm-started network.
    again = PhotorealisticNST(size=48, steps=2, pretrained=False, state_path=state, device="cpu")
    assert again.state is not None and all(torch.equal(again.state[k], nst.state[k]) for k in nst.state)
    # Without warm start nothing is kept or written.
    cold = PhotorealisticNST(size=48, steps=1, pretrained=False, warm_start=False, state_path=tmp_path / "cold.pt", device="cpu")
    cold(content, style)
    assert cold.state is None and not (tmp_path / "cold.pt").exists()


def test_hrnet_training_reduces_the_loss():
    """500 steps per image is the RGCB setting; check a short run actually optimises (random VGG, CPU)."""
    import torch

    from synthinspect.gen.nst_hrnet import STYLE_WEIGHTS, HRNet, VGGReLU, gram

    torch.manual_seed(0)
    vgg = VGGReLU(pretrained=False)
    c, s = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    c_feat = vgg(c)["relu4_2"].detach()
    s_gram = {k: gram(v).detach() for k, v in vgg(s).items() if k in STYLE_WEIGHTS}
    net = HRNet()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    losses = []
    for _ in range(30):
        f = vgg(net(c))
        loss = 150 * torch.mean((f["relu4_2"] - c_feat) ** 2) + 0.5 * sum(
            w * torch.mean((gram(f[k]) - s_gram[k]) ** 2) / (f[k].shape[1] * f[k].shape[2] * f[k].shape[3]) for k, w in STYLE_WEIGHTS.items()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_rgcb_hrnet_saves_state_in_its_workdir(split, bank, tmp_path):
    gen = RGCBGenerator(nst={"size": 48, "steps": 2, "device": "cpu"}, pretrained_vgg=False)
    gen.fit(split, bank, tmp_path / "model")
    host = read_rgb(split.normal_train[1].image)
    rng = np.random.default_rng(5)
    img, label, info = gen.paint(host, bank.sample(host, "stain", rng), bank, rng)
    assert info["nst"] == "hrnet" and info["blend"] == "alpha" and info["scope"] == "global"
    assert (tmp_path / "model" / "nst_hrnet.pt").exists()


def test_rgcb_gatys_option_still_works(split, bank, tmp_path):
    gen = RGCBGenerator(nst_method="gatys", nst={"size": 48, "steps": 2, "device": "cpu"}, pretrained_vgg=False)
    gen.fit(split, bank, tmp_path)
    host = read_rgb(split.normal_train[1].image)
    rng = np.random.default_rng(5)
    img, _, info = gen.paint(host, bank.sample(host, "stain", rng), bank, rng)
    assert img.shape == host.shape and info["nst"] == "gatys"


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
