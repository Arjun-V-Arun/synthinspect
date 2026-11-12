"""Neural style transfer (Gatys et al.), the core of the RGCB approach.

Content is a labelled composite; style is a real image of the same part. The
optimisation starts from the content image, so structure (and therefore the
label) is kept while colour and texture statistics move toward the real image.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# torchvision vgg19.features indices
STYLE_LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1", 29: "relu5_1"}
CONTENT_LAYER = 22  # relu4_2
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeatures(torch.nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        feats = vgg19(weights=weights).features[: max(STYLE_LAYERS) + 1].eval()
        for i, layer in enumerate(feats):
            if isinstance(layer, torch.nn.MaxPool2d):  # Gatys: average pooling gives smoother results
                feats[i] = torch.nn.AvgPool2d(2, 2)
            if isinstance(layer, torch.nn.ReLU):
                feats[i] = torch.nn.ReLU(inplace=False)
        for p in feats.parameters():
            p.requires_grad_(False)
        self.feats = feats

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = (x - _MEAN.to(x)) / _STD.to(x)
        content, styles = None, []
        for i, layer in enumerate(self.feats):
            x = layer(x)
            if i == CONTENT_LAYER:
                content = x
            if i in STYLE_LAYERS:
                styles.append(x)
        return content, styles


def gram(f: torch.Tensor) -> torch.Tensor:
    b, c, h, w = f.shape
    f = f.reshape(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


class StyleTransfer:
    def __init__(
        self,
        size: int = 640,
        steps: int = 150,
        lr: float = 0.02,
        content_weight: float = 1.0,
        style_weight: float = 1e4,
        tv_weight: float = 1e-3,
        device: str | None = None,
        pretrained: bool = True,
    ):
        self.size, self.steps, self.lr = size, steps, lr
        self.cw, self.sw, self.tw = content_weight, style_weight, tv_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = VGGFeatures(pretrained=pretrained).to(self.device)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        img = cv2.resize(img, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(self.device)

    def __call__(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
        h, w = content.shape[:2]
        c = self._to_tensor(content)
        s = self._to_tensor(style)
        with torch.no_grad():
            c_feat, _ = self.net(c)
            _, s_feats = self.net(s)
            s_grams = [gram(f) for f in s_feats]
        x = c.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            xf, x_styles = self.net(x)
            loss = self.cw * F.mse_loss(xf, c_feat)
            loss = loss + self.sw * sum(F.mse_loss(gram(f), g) for f, g in zip(x_styles, s_grams))
            tv = (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
            loss = loss + self.tw * tv
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(0, 1)
        out = (x.detach()[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
