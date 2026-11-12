"""The DefectFill and MAGIC code paths, run end to end on a tiny random UNet (CPU, no downloads).

These check the plumbing: losses are finite, every loss term reaches the LoRA
weights (including the attention loss through gradient checkpointing), adapters
save and reload exactly, guidance maps behave, and generation keeps the
background. They cannot check image quality; that needs the real backbone.
"""

import math

import numpy as np
import pytest
import torch

from fixtures import tiny_components
from synthinspect.data.maskbank import MaskBank
from synthinspect.gen import sd_inpaint as sdi
from synthinspect.gen.sd_generators import DefectFillGenerator, MagicGenerator
from synthinspect.io import read_mask, read_rgb


@pytest.fixture()
def comp():
    return tiny_components(32)


def _data(split, dtype="stain"):
    srcs = [s for s in split.defect_train if s.defect_type == dtype]
    return [read_rgb(s.image) for s in srcs], [read_mask(s.mask) for s in srcs]


def test_placeholder_position(comp):
    assert sdi.placeholder_positions(comp, "A photo of sks") == [4]  # <bos> a photo of sks


@pytest.mark.parametrize("checkpointing", [False, True])
def test_defectfill_every_loss_reaches_lora(comp, split, checkpointing):
    images, masks = _data(split)
    cfg = sdi.TrainConfig(objective="defectfill", steps=1, gradient_checkpointing=checkpointing, log_every=0)
    # Run one step by hand so we can inspect the gradients before the optimizer zeroes them.
    sdi.add_lora(comp.unet, cfg.rank, sdi.UNET_TARGETS, "probe")
    sdi.set_active(comp.unet, "probe")
    params = sdi.lora_params(comp.unet, "probe")
    if checkpointing:
        comp.unet.enable_gradient_checkpointing()
    comp.unet.train()
    store = sdi.install_attention_capture(comp.unet)
    x, m = sdi.to_model_space(images[0], masks[0], comp.resolution)
    x, m = x[None], m[None]
    lat = comp.resolution // comp.latent_factor
    latents = sdi._encode(comp, x)
    m_lat = sdi._latent_mask(m, lat)
    masked = sdi._encode(comp, x * (1 - m))
    store.reset(sdi.placeholder_positions(comp, "A photo of sks"))
    enc = sdi.encode_text(comp, ["A photo of sks"])
    pred = comp.unet(torch.cat([latents, m_lat, masked], 1), torch.tensor([10]), encoder_hidden_states=enc).sample
    amap = sdi.attention_map(store, lat)
    assert amap.shape == (1, lat, lat) and float(amap.max()) == pytest.approx(1.0)
    l_attn = torch.nn.functional.mse_loss(amap, m_lat[:, 0])
    l_attn.backward()
    grad = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
    assert grad > 0, "attention loss did not reach the LoRA weights"


def test_defectfill_training_runs_and_adapter_roundtrips(comp, split):
    images, masks = _data(split)
    cfg = sdi.TrainConfig(objective="defectfill", steps=3, log_every=0)
    state = sdi.train_lora(comp, images, masks, "widget", "d_stain", cfg)
    assert state["unet"] and state["text"]  # both UNet and text encoder adapters
    # LoRA B matrices start at zero; non-zero after training means gradients really reached both
    # adapters with gradient checkpointing on (a reentrant checkpoint would silently drop them).
    for part in ("unet", "text"):
        b = [v for k, v in state[part].items() if "lora_B" in k]
        assert b and any(float(v.abs().sum()) > 0 for v in b), f"{part} LoRA did not train"
    assert all(np.isfinite(list(h.values())).all() for h in state["history"])
    assert set(state["history"][0]) == {"def", "obj", "attn"}
    # Reload into a fresh copy and check the weights are identical.
    fresh = tiny_components(32)
    sdi.add_lora(fresh.unet, cfg.rank, sdi.UNET_TARGETS, "d_stain")
    sdi.load_lora_state(fresh.unet, state["unet"])
    for k, v in state["unet"].items():
        assert torch.equal(fresh.unet.state_dict()[k], v)


def test_magic_training_uses_unet_only(comp, split):
    images, masks = _data(split, "scratch")
    cfg = sdi.TrainConfig(objective="magic", steps=2, log_every=0)
    state = sdi.train_lora(comp, images, masks, "widget", "d_scratch", cfg)
    assert state["unet"] and not state["text"]
    assert set(state["history"][0]) == {"mse"}


def test_magic_guidance_ramps_up_inside_mask():
    first = sdi.magic_guidance(0, 30)
    last = sdi.magic_guidance(29, 30)
    assert first == (pytest.approx(4.0), 7.5)
    assert last == (pytest.approx(7.5), 7.5)
    mids = [sdi.magic_guidance(i, 30)[0] for i in range(30)]
    assert all(b >= a for a, b in zip(mids, mids[1:]))


def test_inpaint_shapes_and_seed(comp, split):
    img, mask = _data(split)[0][0], _data(split)[1][0]
    a = sdi.inpaint(comp, img, mask, "A photo of sks", n=2, steps=3, guidance=sdi.constant_guidance(3.0), seed=5)
    b = sdi.inpaint(comp, img, mask, "A photo of sks", n=2, steps=3, guidance=sdi.constant_guidance(3.0), seed=5)
    assert len(a) == 2 and a[0].shape == (32, 32, 3) and a[0].dtype == np.uint8
    assert all((x == y).all() for x, y in zip(a, b))


def test_composite_keeps_background(split):
    host = read_rgb(split.normal_train[0].image)
    mask = np.zeros(host.shape[:2], bool)
    mask[40:50, 40:50] = True
    gen = np.zeros((32, 32, 3), np.uint8)
    out = sdi.composite(host, gen, mask, feather=2, context=2)
    assert (out[:30] == host[:30]).all()
    assert out[45, 45].sum() < host[45, 45].sum()


@pytest.mark.parametrize("cls", [DefectFillGenerator, MagicGenerator])
def test_generators_fit_and_paint(cls, split, tmp_path):
    comp = tiny_components(32)
    kw = dict(train={"steps": 1, "log_every": 0}, sample_steps=2, components=comp)
    if cls is DefectFillGenerator:
        kw.update(lfs_candidates=3, lpips_fn=lambda a, b: float(np.abs(a.astype(int) - b.astype(int)).mean()))
    gen = cls("widget", **kw)
    bank = MaskBank(split.defect_train)
    gen.fit(split, bank, tmp_path)
    assert gen.adapter_path("stain").exists() and gen.adapter_path("scratch").exists()
    host = read_rgb(split.normal_train[2].image)
    rng = np.random.default_rng(0)
    placed = bank.sample(host, "stain", rng)
    img, label, info = gen.paint(host, placed, bank, rng)
    assert img.shape == host.shape and (label == placed.mask).all()
    if cls is DefectFillGenerator:
        assert len(info["lfs_scores"]) == 3 and info["lfs_choice"] == int(np.argmax(info["lfs_scores"]))
    # A second generator instance loads the saved adapters from disk instead of retraining.
    gen2 = cls("widget", **{**kw, "components": tiny_components(32)})
    gen2.fit(split, bank, tmp_path)
    gen2.paint(host, placed, bank, np.random.default_rng(0))
