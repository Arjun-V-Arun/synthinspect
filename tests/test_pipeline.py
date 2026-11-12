"""The whole experiment, end to end, on the tiny fixture: every stage the README tells you to run.

Detectors train for two epochs on 128-pixel images, so the numbers mean nothing;
the test proves the stages connect, resume, and produce the files the report,
the API and the browser demo read.
"""

import json
import shutil

import pytest
import yaml

from synthinspect.cli import main


@pytest.fixture(scope="module")
def smoke(tmp_path_factory, mvtec_root):
    root = tmp_path_factory.mktemp("smoke")
    (root / "configs").mkdir()
    exp = {
        "mvtec_root": str(mvtec_root),
        "runs_dir": str(root / "runs"),
        "categories": {"widget": {"placement": {"rotate": "free", "flip": True}, "yolo_aug": {"flipud": 0.5}}},
        "split": {"train_fraction_den": 3},
        "generation": {"n_per_type": 4, "seed": 0, "methods": ["copypaste", "rgcb"]},
        "dataset": {"class_mode": "binary", "n_normals": 12},
        "detectors": {"yolo_s": {"family": "yolo", "model": "yolo26n", "pretrained": False, "epochs": 2, "imgsz": 128, "batch": 8, "workers": 0}},
        "suites": {
            "core": {"conditions": ["real", "copypaste", "rgcb"], "detectors": ["yolo_s"], "seeds": [0, 1]},
            "models": {"conditions": ["real", "best"], "detectors": ["yolo_s"], "seeds": [0]},
        },
        "evaluation": {"n_boot": 50, "box_min_area": 9, "box_merge_frac": 0.01},
        "review": {"max_escape": 0.0, "max_false_reject": 0.05, "repeats": 3},
        "xai": {"odam_top_k": 1, "pointing_tolerance": 5, "drise": {"n_masks": 40, "grid": 6, "p": 0.5, "batch": 8}, "drise_images": 2, "faithfulness_images": 2},
        "export": {"calibration_limit": 10, "bench_runs": 3, "bench_warmup": 1, "int8_max_ap_drop": 1.0},
    }
    gens = {"copypaste": {"blend": "poisson"}, "rgcb": {"scope": "global", "pretrained_vgg": False, "nst": {"size": 48, "steps": 2, "device": "cpu"}}}
    (root / "configs" / "experiment.yaml").write_text(yaml.safe_dump(exp))
    (root / "configs" / "generators.yaml").write_text(yaml.safe_dump(gens))
    return root


def test_full_pipeline(smoke, monkeypatch):
    cfg = str(smoke / "configs" / "experiment.yaml")
    run = lambda *a: main(["--config", cfg, *a])  # noqa: E731
    assert run("prepare") == 0
    for m in ("copypaste", "rgcb"):
        assert run("generate", "--category", "widget", "--method", m) == 0
    assert run("train", "--suite", "core", "--device", "cpu") == 0
    assert run("evaluate", "--suite", "core") == 0
    runs = smoke / "runs" / "widget"
    res = json.loads((runs / "results" / "rgcb" / "yolo_s" / "s1.json").read_text())
    assert {"image_ap", "image_auroc", "box_ap10", "box_ap50"} <= set(res["metrics"])
    assert 0 <= res["review"]["automation"] <= 1
    assert run("compare") == 0
    comps = json.loads((smoke / "runs" / "results" / "comparisons.json").read_text())
    assert any("copypaste-vs-rgcb" in k for k in comps) and any("-vs-real" in k for k in comps)
    assert run("train", "--suite", "models", "--device", "cpu") == 0  # resolves "best" from the core results
    assert run("evaluate", "--suite", "models") == 0
    assert run("export", "--category", "widget", "--detector", "yolo_s") == 0
    exp_reports = list((runs / "export").glob("*/yolo_s/s0/report.json"))
    rep = json.loads(exp_reports[0].read_text())
    assert {"torch_fp32", "onnx_fp32", "openvino_fp16", "openvino_int8", "onnx_int8"} <= set(rep["runtimes"])
    assert rep["runtimes"]["openvino_int8"]["size_mb"] < rep["runtimes"]["onnx_fp32"]["size_mb"]
    assert run("explain", "--category", "widget", "--detector", "yolo_s") == 0
    summary = json.loads(next((runs / "xai").glob("yolo_s/s0/summary.json")).read_text())
    assert set(summary["localisation"]) == {"real", "copypaste", "rgcb"}
    assert "quantisation" in summary and "faithfulness" in summary

    # Packaging writes into the repo's web/ folder; point it at a temp copy instead.
    import synthinspect.pipeline as P

    fake_repo = smoke / "repo"
    (fake_repo / "web").mkdir(parents=True)
    (fake_repo / "README.md").write_text("x\n<!-- results:start -->\n<!-- results:end -->\n")
    monkeypatch.setattr(P, "REPO", fake_repo)
    assert run("report") == 0
    text = (smoke / "runs" / "results" / "RESULTS.md").read_text()
    assert "Generator Comparison" in text and "Ship Small" in text and "Did INT8 Change" in text
    assert "RGCB (copy-paste + NST)" in (fake_repo / "README.md").read_text()
    assert run("demo", "--category", "widget", "--detector", "yolo_s", "--n-samples", "4") == 0
    web = fake_repo / "web" / "assets"
    cfgj = json.loads((web / "config.json").read_text())
    assert (web / "model.onnx").exists() and len(cfgj["samples"]) == 4
    assert cfgj["band"]["t_pass"] <= cfgj["band"]["t_reject"]

    # The API serves the packaged model.
    from fastapi.testclient import TestClient

    import synthinspect.serve.app as app_mod

    monkeypatch.setenv("SYNTHINSPECT_MODEL_DIR", str(runs / "package"))
    monkeypatch.setattr(app_mod, "_service", None)
    client = TestClient(app_mod.app)
    assert client.get("/readyz").json()["runtime"] == "openvino-int8"
    img = (runs / "package" / "samples" / "00.jpg").read_bytes()
    r = client.post("/v1/inspect", files={"file": ("a.jpg", img, "image/jpeg")}).json()
    assert r["decision"] in {"pass", "review", "reject"} and r["model_version"] == cfgj["version"]
    assert client.post("/v1/inspect", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 400
    m = client.get("/metrics").text
    assert 'inspect_decisions_total{decision="' in m and "inspect_latency_ms_count 1" in m
    card = client.get("/v1/model").json()
    assert card["condition"] in {"copypaste", "rgcb"} and "license" in card
