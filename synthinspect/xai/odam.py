"""ODAM-style explanations (Zhao et al., ICLR 2023) for YOLO26's NMS-free head.

Plain Grad-CAM averages gradients over a whole feature map and explains one
class score for one image, which blurs together every object a detector sees.
ODAM keeps the gradient per location: the map for one detection is

    H = ReLU( sum over channels of (dScore/dA * A) )

computed at each feature level the head reads (P3, P4, P5), upsampled and
summed. The target is the score of one specific detection, so two defects on
the same part get two different maps. YOLO26's one-to-one head selects its 300
outputs with top-k, which passes gradients to the chosen scores, so one
backward pass per detection is enough.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..detect.base import Letterbox


def _unletterbox(sal: np.ndarray, lb: Letterbox, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    nh, nw = round(h * lb.ratio), round(w * lb.ratio)
    crop = sal[int(lb.pad_y) : int(lb.pad_y) + nh, int(lb.pad_x) : int(lb.pad_x) + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def odam(det, image: np.ndarray, indices: list[int] | None = None, top_k: int = 1) -> tuple[list[np.ndarray], np.ndarray]:
    """Explain detections of one image.

    Returns (saliency maps in original image size, normalised to [0, 1]; raw (300, 6) output in letterboxed pixels).
    ``indices`` are rows of the raw output (sorted by score); default: the top ``top_k``.
    """
    feats: dict[str, list[torch.Tensor]] = {}

    def hook(module, inputs, output):
        maps = list(inputs[0])
        for m in maps:
            m.retain_grad()
        feats["maps"] = maps

    handle = det.head.register_forward_hook(hook)
    try:
        x, infos = det.preprocess([image])
        x.requires_grad_(True)  # inference weights are frozen; this makes the feature maps part of the graph
        with torch.enable_grad():
            y = det.forward(x)  # (1, 300, 6)
            raw = y.detach().float().cpu().numpy()[0]
            idx = indices if indices is not None else list(range(top_k))
            out = []
            for k, i in enumerate(idx):
                for m in feats["maps"]:
                    m.grad = None
                y[0, i, 4].backward(retain_graph=k < len(idx) - 1)
                total = torch.zeros(det.imgsz, det.imgsz, device=x.device)
                for m in feats["maps"]:
                    h = F.relu((m.grad * m).sum(1, keepdim=True))
                    total += F.interpolate(h, size=(det.imgsz, det.imgsz), mode="bilinear", align_corners=False)[0, 0]
                sal = _unletterbox(total.detach().cpu().numpy(), infos[0], image.shape[:2])
                out.append(normalise(sal))
    finally:
        handle.remove()
    return out, raw


def normalise(sal: np.ndarray) -> np.ndarray:
    sal = np.maximum(sal, 0).astype(np.float32)
    m = sal.max()
    return sal / m if m > 0 else sal
