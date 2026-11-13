"""Photorealistic style transfer with a small HRNet, as used in the RGCB generation pipeline.

The microbial-colony generator used at RGCB (Pawłowski, Majchrowska and Golan,
Scientific Reports 2022; github.com/jarek-pawlowski/microbial-dataset-generation)
does not optimise pixels like Gatys et al. It trains a small high-resolution
network (Li, Ye and Li, 2019, "High-resolution network for photorealistic style
transfer"; MIT-licensed reference code by Ming Li) to map the content image to a
restyled one. Keeping the full-resolution branch throughout is what preserves
fine structure, which is why it suits labelled synthetic images.

Settings below follow that generator:
  losses on VGG-19 ReLU features: content relu4_2; style relu1_1..relu5_1
  layer weights 0.1, 0.2, 0.4, 0.8, 1.6; unnormalised Gram matrices, each layer's
  error divided by c*h*w; total = 150 * content + 0.5 * style
  Adam lr 5e-3, step decay x0.9 every 200 steps, 500 steps per image
  warm start: each image starts from the network trained on the previous one

This file is a re-implementation from the paper and those settings, not a copy.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# torchvision vgg19.features: index of the ReLU after each named conv
LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1", 22: "relu4_2", 29: "relu5_1"}
STYLE_WEIGHTS = {"relu1_1": 0.1, "relu2_1": 0.2, "relu3_1": 0.4, "relu4_1": 0.8, "relu5_1": 1.6}


class Bottleneck(nn.Module):
    """1x1 -> kxk -> 1x1 convolutions with instance norm, plus a residual connection."""

    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__()
        mid = cout // 4
        self.body = nn.Sequential(
            nn.Conv2d(cin, mid, 1), nn.InstanceNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, mid, k, padding=k // 2), nn.InstanceNorm2d(mid), nn.ReLU(),
            nn.Conv2d(mid, cout, 1), nn.InstanceNorm2d(cout), nn.ReLU(),
        )
        self.skip = None if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.InstanceNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + (x if self.skip is None else self.skip(x)))


class HRNet(nn.Module):
    """Three resolution branches (1, 1/2, 1/4) fused back into the full-resolution branch."""

    def __init__(self):
        super().__init__()
        self.b1_1, self.b1_2, self.b1_3 = Bottleneck(3, 16), Bottleneck(16, 32), Bottleneck(32, 32)
        self.b1_4, self.b1_5 = Bottleneck(64, 64), Bottleneck(192, 64)
        self.b1_6, self.b1_7 = Bottleneck(64, 32), Bottleneck(32, 16)
        self.out = nn.Conv2d(16, 3, 3, padding=1)
        self.b2_1 = Bottleneck(32, 32)
        self.down1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.down1_2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.down1_4 = nn.Conv2d(32, 32, 3, stride=4, padding=1)
        self.down2_1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

    @staticmethod
    def _up(x: torch.Tensor, f: int) -> torch.Tensor:
        return F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m1 = self.b1_1(x)  # full res, 16
        m2 = self.b1_2(m1)  # full res, 32
        m3 = self.down1_1(m1)  # 1/2, 32
        m4 = torch.cat([self.b1_3(m2), self._up(m3, 2)], 1)  # full, 64
        m5 = torch.cat([self.down1_2(m2), self.b2_1(m3)], 1)  # 1/2, 64
        m6 = torch.cat([self.down1_4(m2), self.down2_1(m3)], 1)  # 1/4, 64
        m7 = torch.cat([self.b1_4(m4), self._up(m5, 2), self._up(m6, 4)], 1)  # full, 192
        return self.out(self.b1_7(self.b1_6(self.b1_5(m7))))


class VGGReLU(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features[: max(LAYERS) + 1].eval()
        for i, layer in enumerate(feats):
            if isinstance(layer, nn.ReLU):
                feats[i] = nn.ReLU(inplace=False)
        for p in feats.parameters():
            p.requires_grad_(False)
        self.feats = feats

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for i, layer in enumerate(self.feats):
            x = layer(x)
            if i in LAYERS:
                out[LAYERS[i]] = x
        return out


def gram(f: torch.Tensor) -> torch.Tensor:
    b, c, h, w = f.shape
    f = f.reshape(b * c, h * w)
    return f @ f.t()  # unnormalised, as in the RGCB generator


class PhotorealisticNST:
    """Restyle ``content`` toward ``style`` by training a small HRNet on the VGG losses.

    ``amp`` runs the convolutions (VGG loss network and HRNet) in bfloat16 for speed.
    It is not an RGCB setting: Gram matrices and losses stay in float32, and the
    returned image comes from a float32 pass of the trained network, so the output
    should match ``amp: false`` closely (``scripts/compare_gen.py`` measures it).

    ``warm_steps`` (not an RGCB setting) runs fewer steps once the network is
    warm-started from the previous image; the first, cold image still gets ``steps``.

    After each call ``last_trace`` holds the loss at a few steps and how much the
    output would have differed (mean absolute, 0-255) had optimisation stopped at
    each of ``trace_steps``: the evidence for or against cutting the step count.
    """

    def __init__(
        self,
        size: int = 640,
        steps: int = 500,
        lr: float = 5e-3,
        lr_step: int = 200,
        lr_gamma: float = 0.9,
        content_weight: float = 150.0,
        style_weight: float = 0.5,
        warm_start: bool = True,
        state_path: str | Path | None = None,
        device: str | None = None,
        pretrained: bool = True,
        amp: bool = True,
        warm_steps: int | None = None,
        trace_steps: tuple[int, ...] = (25, 50, 100, 200),
    ):
        self.size, self.steps, self.lr, self.lr_step, self.lr_gamma = size, steps, lr, lr_step, lr_gamma
        self.cw, self.sw = content_weight, style_weight
        self.warm_start = warm_start
        self.warm_steps = int(warm_steps) if warm_steps else None
        self.state_path = Path(state_path) if state_path else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # fixed input size per category: let cuDNN pick the fastest kernels
        self.amp = bool(amp) and (self.device.type == "cpu" or (self.device.type == "cuda" and torch.cuda.is_bf16_supported()))
        self.trace_steps = tuple(sorted(int(k) for k in trace_steps))
        self.vgg = VGGReLU(pretrained).to(self.device)
        if self.amp:
            self.vgg = self.vgg.to(memory_format=torch.channels_last)  # faster bf16 convolutions
        self.state: dict | None = None
        self.last_trace: dict = {}
        if warm_start and self.state_path and self.state_path.exists():
            self.state = torch.load(self.state_path, map_location="cpu", weights_only=True)

    def _tensor(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nh, nw = max(4, round(h * s)), max(4, round(w * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(self.device)
        return (t - _MEAN.to(t)) / _STD.to(t)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp)

    def _features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """VGG features, computed in bf16 under ``amp`` and handed back as float32."""
        if self.amp:
            x = x.contiguous(memory_format=torch.channels_last)
        with self._autocast():
            f = self.vgg(x)
        return {k: v.float() for k, v in f.items()}

    @staticmethod
    def _pixels(t: torch.Tensor) -> torch.Tensor:
        return (t * _STD.to(t) + _MEAN.to(t)).clamp(0, 1) * 255

    def __call__(self, content: np.ndarray, style: np.ndarray, seed: int | None = None) -> np.ndarray:
        h0, w0 = content.shape[:2]
        c = self._tensor(content)
        s = self._tensor(style)
        H, W = c.shape[2], c.shape[3]
        # HRNet needs sides divisible by 4: reflect-pad, crop after
        ph, pw = (-H) % 4, (-W) % 4
        c_pad = F.pad(c, (0, pw, 0, ph), mode="reflect") if (ph or pw) else c
        with torch.no_grad():  # targets go through the same precision as the output does
            c_feat = self._features(c)["relu4_2"]
            s_grams = {k: gram(v) for k, v in self._features(s).items() if k in STYLE_WEIGHTS}

        cold = not (self.warm_start and self.state is not None)
        if cold and seed is not None:
            with torch.random.fork_rng(devices=[]):  # seeded first network, without touching anyone else's RNG
                torch.manual_seed(seed)
                net = HRNet()
        else:
            net = HRNet()
        net = net.to(self.device)
        if not cold:
            net.load_state_dict(self.state)
        steps = self.warm_steps if (not cold and self.warm_steps) else self.steps
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_step, gamma=self.lr_gamma)

        # The image returned after k steps is the network's output before its k-th update.
        record = {k for k in self.trace_steps if k < steps} | {steps}
        snaps: dict[int, torch.Tensor] = {}
        losses: dict[str, float] = {}
        for i in range(steps):
            k = i + 1
            with self._autocast():
                out = net(c_pad)[..., :H, :W]
            out = out.float()
            if k in record:
                if self.amp:  # a float32 pass, so the kept image carries no bf16 rounding
                    with torch.no_grad():
                        snaps[k] = net(c_pad)[..., :H, :W].float()
                else:
                    snaps[k] = out.detach().clone()
            f = self._features(out)
            content_loss = torch.mean((f["relu4_2"] - c_feat) ** 2)
            style_loss = 0.0
            for name, wgt in STYLE_WEIGHTS.items():
                _, ch, hh, ww = f[name].shape
                style_loss = style_loss + wgt * torch.mean((gram(f[name]) - s_grams[name]) ** 2) / (ch * hh * ww)
            loss = self.cw * content_loss + self.sw * style_loss
            if k == 1 or k in record:
                losses[str(k)] = round(float(loss.detach()), 4)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        final = self._pixels(snaps[steps])
        self.last_trace = {
            "cold": cold,
            "amp": self.amp,
            "steps": steps,
            "loss": losses,
            # mean absolute difference (0-255) from the final image, had we stopped after k steps
            "change_after": {str(k): round(float((self._pixels(v) - final).abs().mean()), 3) for k, v in sorted(snaps.items()) if k != steps},
        }
        if self.warm_start:
            self.state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if self.state_path:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.state_path.with_suffix(".tmp")
                torch.save(self.state, tmp)
                tmp.replace(self.state_path)
        img = (final / 255)[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).round().astype(np.uint8)
        return cv2.resize(img, (w0, h0), interpolation=cv2.INTER_CUBIC)
