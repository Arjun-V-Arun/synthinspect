"""Package one exported run for serving and for the browser demo.

Writes  runs/<cat>/package/   (config.json, model.onnx, openvino/int8, samples)
and copies the browser parts to web/assets/.

The demo thresholds are fitted on the whole test set, because it is the only
labelled real data. They are display thresholds for the demo; the honest
operating numbers are the cross-fitted ones in RESULTS.md, and a real line
would calibrate on its own held-out parts (docs/PRODUCTION.md).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from ..eval.review import fit_band
from ..io import file_sha256, read_json, read_rgb, write_json, write_rgb


def package_demo(cfg: dict, cat: str, cond: str | None, detector: str, seed: int, n_samples: int = 8) -> Path:
    from .. import pipeline as P
    from ..detect.yolo import TorchYOLO
    from ..xai.odam import odam

    cond = cond or P.best_generator(cfg, cat)
    exp = P.cat_dir(cfg, cat) / "export" / cond / detector / f"s{seed}"
    rep_path = exp / "report.json"
    if not rep_path.exists():
        raise FileNotFoundError(f"Run `export --category {cat} --condition {cond}` first")
    rep = read_json(rep_path)
    if cfg["detectors"][detector]["family"] != "yolo":
        raise ValueError("The browser demo decodes YOLO26's NMS-free output; package a YOLO26 run")

    out = P.cat_dir(cfg, cat) / "package"
    if out.exists():
        shutil.rmtree(out)
    (out / "samples").mkdir(parents=True)
    shutil.copy(exp / "model.onnx", out / "model.onnx")
    shutil.copytree(exp / "openvino" / "int8", out / "openvino" / "int8")
    web_model = exp / "model.int8.onnx" if (exp / "model.int8.onnx").exists() else exp / "model.fp16.onnx"
    shutil.copy(web_model, out / "model.web.onnx")

    preds = P.load_preds(P.cat_dir(cfg, cat) / "preds" / cond / detector / f"s{seed}.json")
    sp = P.load_split(cfg, cat)
    labels = np.array([int(s.is_defect) for s in sp.test])
    scores = np.array([p.image_score for p in preds])
    band = fit_band(scores, labels, cfg["review"]["max_escape"], cfg["review"]["max_false_reject"])
    res = read_json(P.cat_dir(cfg, cat) / "results" / cond / detector / f"s{seed}.json")

    # Samples: a mix of good and defective parts, with a precomputed ODAM explanation each.
    rng = np.random.default_rng(seed)
    good = [i for i, s in enumerate(sp.test) if not s.is_defect]
    bad = [i for i, s in enumerate(sp.test) if s.is_defect]
    pick = list(rng.choice(good, min(len(good), n_samples // 4), replace=False)) + list(rng.choice(bad, min(len(bad), n_samples - n_samples // 4), replace=False))
    det = TorchYOLO(P.weights_of(cfg, cat, cond, detector, seed), rep["imgsz"])
    samples = []
    for k, i in enumerate(pick):
        s = sp.test[int(i)]
        img = read_rgb(s.image)
        small = _fit(img, 640)
        write_rgb(out / "samples" / f"{k:02d}.jpg", small)
        maps, _ = odam(det, img, indices=[0])
        heat = cv2.applyColorMap((cv2.resize(maps[0], small.shape[1::-1]) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)
        cv2.imwrite(str(out / "samples" / f"{k:02d}_odam.jpg"), overlay)
        samples.append({"file": f"{k:02d}.jpg", "explanation": f"{k:02d}_odam.jpg", "label": "defect" if s.is_defect else "good", "defect_type": s.defect_type})

    good_scores = scores[labels == 0].tolist()
    config = {
        "category": cat,
        "condition": cond,
        "detector": detector,
        "seed": seed,
        "format": rep["format"],
        "imgsz": rep["imgsz"],
        "band": band.to_dict(),
        "version": file_sha256(out / "model.onnx")[:12],
        "metrics": {k: v["value"] for k, v in res["metrics"].items()} | {"review_cross_fitted": res["review"]},
        "export": {k: {kk: v[kk] for kk in ("size_mb", "p50_ms", "image_ap", "d_image_ap")} for k, v in rep["runtimes"].items()},
        "reference_scores": [round(x, 5) for x in scores.tolist()],
        "good_reference_scores": [round(x, 5) for x in good_scores],
        "samples": samples,
        "license": "Images: MVTec AD, CC BY-NC-SA 4.0 (non-commercial). Model trained on MVTec AD; same terms.",
    }
    write_json(out / "config.json", config)

    web = P.REPO / "web" / "assets"
    if web.exists():
        shutil.rmtree(web)
    (web / "samples").mkdir(parents=True)
    shutil.copy(out / "model.web.onnx", web / "model.onnx")
    for f in (out / "samples").iterdir():
        shutil.copy(f, web / "samples" / f.name)
    write_json(web / "config.json", {k: config[k] for k in ("category", "condition", "detector", "format", "imgsz", "band", "version", "metrics", "samples", "license")})
    P._log(f"packaged {out} and {web}")
    return out


def _fit(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
