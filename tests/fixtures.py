"""Tiny MVTec-shaped dataset and tiny diffusion components, so every test runs on CPU with no downloads."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from synthinspect.io import write_mask, write_rgb


def _part(rng: np.random.Generator, size: int) -> np.ndarray:
    img = np.full((size, size, 3), 40, np.uint8)  # dark background, like metal_nut
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)
    c = size // 2 + int(rng.integers(-size // 20, size // 20 + 1))
    cv2.circle(img, (c, c), int(size * 0.32), (170, 170, 175), -1)
    cv2.circle(img, (c, c), int(size * 0.12), (40, 40, 40), -1)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    mask = np.zeros(img.shape[:2], np.uint8)
    out = img.copy()
    c = size // 2
    if kind == "scratch":
        a = rng.uniform(0, np.pi)
        r = size * rng.uniform(0.15, 0.28)
        p1 = (int(c + r * np.cos(a)), int(c + r * np.sin(a)))
        p2 = (int(c + r * np.cos(a) + size * 0.1), int(c + r * np.sin(a) + size * 0.04))
        cv2.line(out, p1, p2, (240, 240, 240), max(2, size // 64))
        cv2.line(mask, p1, p2, 255, max(2, size // 64) + 2)
    else:  # "stain"
        a = rng.uniform(0, 2 * np.pi)
        r = size * 0.22
        p = (int(c + r * np.cos(a)), int(c + r * np.sin(a)))
        cv2.circle(out, p, size // 16, (110, 70, 40), -1)
        cv2.circle(mask, p, size // 16 + 1, 255, -1)
    return out, mask > 0


def make_mvtec(root: Path, category: str = "widget", size: int = 128, n_train: int = 24, n_test_good: int = 6, n_per_type: int = 9, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    cat = root / category
    for i in range(n_train):
        write_rgb(cat / "train" / "good" / f"{i:03d}.png", _part(rng, size))
    for i in range(n_test_good):
        write_rgb(cat / "test" / "good" / f"{i:03d}.png", _part(rng, size))
    for kind in ("scratch", "stain"):
        for i in range(n_per_type):
            img, mask = _defect(_part(rng, size), kind, rng)
            write_rgb(cat / "test" / kind / f"{i:03d}.png", img)
            write_mask(cat / "ground_truth" / kind / f"{i:03d}_mask.png", mask)
    return root


# ----------------------------------------------------------------------------- tiny diffusion parts


class TinyTokenizer:
    """Whitespace tokenizer with the CLIPTokenizer call signature used by sd_inpaint."""

    model_max_length = 8

    def __init__(self):
        self.vocab = {"<bos>": 0, "<eos>": 1, "<pad>": 2}

    def _ids(self, text: str) -> list[int]:
        out = []
        for w in text.lower().split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
            out.append(self.vocab[w])
        return out

    def __call__(self, text, padding=None, max_length=None, truncation=False, return_tensors=None, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else list(text)
        rows = []
        for t in texts:
            ids = self._ids(t)
            if add_special_tokens:
                ids = [0] + ids + [1]
            if max_length:
                ids = ids[:max_length]
                if padding == "max_length":
                    ids = ids + [2] * (max_length - len(ids))
            rows.append(ids)

        class Out:
            pass

        o = Out()
        if return_tensors == "pt":
            o.input_ids = torch.tensor(rows)
        else:
            o.input_ids = rows[0] if isinstance(text, str) else rows
        return o


def tiny_components(resolution: int = 32):
    from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel

    from synthinspect.gen.sd_inpaint import SDComponents

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=resolution // 2,
        in_channels=9,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        latent_channels=4,
        norm_num_groups=8,
        layers_per_block=1,
    )
    te = CLIPTextModel(
        CLIPTextConfig(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=8)
    )
    for m in (unet, vae, te):
        m.requires_grad_(False)
    sched = DDPMScheduler(num_train_timesteps=100)
    return SDComponents(
        unet=unet,
        vae=vae,
        text_encoder=te,
        tokenizer=TinyTokenizer(),
        train_scheduler=sched,
        sample_scheduler=DDIMScheduler.from_config(sched.config),
        device=torch.device("cpu"),
        dtype=torch.float32,
        resolution=resolution,
    )
