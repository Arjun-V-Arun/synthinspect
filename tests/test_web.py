"""The browser demo's JavaScript gives the same answers as the Python pipeline.

Runs web/inspect.js under Node with onnxruntime-web, the WASM runtime the
browser uses, on an ONNX model exported from the fixture detector.
Skipped when Node or web/node_modules is missing (run `npm install` in web/).
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

WEB = Path(__file__).resolve().parents[1] / "web"
pytestmark = pytest.mark.skipif(shutil.which("node") is None or not (WEB / "node_modules" / "onnxruntime-web").exists(), reason="needs node and npm install in web/")


@pytest.fixture(scope="module")
def onnx_model(trained_yolo, tmp_path_factory):
    from synthinspect.deploy.export import export_yolo_onnx

    return export_yolo_onnx(trained_yolo, tmp_path_factory.mktemp("web") / "model.onnx", imgsz=128)


def test_js_matches_python(onnx_model, split, tmp_path):
    from synthinspect.detect.base import letterbox
    from synthinspect.detect.runtime import OnnxDetector, preprocess
    from synthinspect.eval.review import Band
    from synthinspect.io import read_rgb

    img = read_rgb(next(s for s in split.test if s.is_defect).image)
    x, _ = preprocess([img], 128, "yolo26")
    py = OnnxDetector(onnx_model, 128).predict([img], min_score=0.001)[0].top(10)
    band = Band(0.3, 0.6)
    sizes = [(900, 900), (1024, 768), (641, 333), (333, 641), (700, 700)]
    job = {
        "model": str(onnx_model),
        "size": 128,
        "tensor": x.ravel().tolist(),
        "origW": img.shape[1],
        "origH": img.shape[0],
        "band": band.to_dict(),
        "sizes": sizes,
        "drise_n": 8,
    }
    (tmp_path / "job.json").write_text(json.dumps(job))
    out = json.loads(subprocess.run(["node", str(WEB / "check.mjs"), str(tmp_path / "job.json")], capture_output=True, text=True, check=True, cwd=WEB).stdout)

    js_scores = np.array([d["score"] for d in out["dets"]])
    js_boxes = np.array([d["box"] for d in out["dets"]])
    n = min(len(js_scores), len(py.scores), 5)
    assert np.allclose(js_scores[:n], py.scores[:n], atol=1e-4)
    top_py, top_js = py.boxes[0], js_boxes[0]
    assert np.allclose(top_js, top_py, atol=0.5)
    assert out["decision"] == {0: "pass", 1: "review", 2: "reject"}[int(band.decide(np.array([py.image_score]))[0])]
    # Letterbox geometry agrees with the Python letterbox for awkward sizes (Python rounds halves to even).
    for (w, h), geo in zip(sizes, out["geometries"]):
        _, lb = letterbox(np.zeros((h, w, 3), np.uint8), 128)
        assert (geo["padX"], geo["padY"]) == (lb.pad_x, lb.pad_y)
        assert geo["ratio"] == pytest.approx(lb.ratio)
    assert out["drise"]["length"] == 128 * 128 and out["drise"]["max"] == pytest.approx(1.0)
