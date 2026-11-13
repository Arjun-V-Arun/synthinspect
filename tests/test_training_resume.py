"""A training run counts as finished only when done.json exists; an interrupted YOLO run resumes.

Ultralytics saves last.pt after every epoch, so before this marker existed an
interrupted overnight run left a half-trained checkpoint that the next run (and
the evaluation) would have treated as final.
"""

import pytest
import yaml

import synthinspect.pipeline as P
from synthinspect.cli import main


class SimulatedCrash(RuntimeError):
    pass


@pytest.fixture()
def cfg(tmp_path, mvtec_root):
    exp = {
        "mvtec_root": str(mvtec_root),
        "runs_dir": str(tmp_path / "runs"),
        "categories": {"widget": {"placement": {"rotate": "free", "flip": True}}},
        "split": {"train_fraction_den": 3},
        "generation": {"n_per_type": 2, "seed": 0, "methods": ["copypaste"]},
        "dataset": {"class_mode": "binary", "n_normals": 8},
        "detectors": {"yolo_s": {"family": "yolo", "model": "yolo26n", "pretrained": False, "epochs": 3, "imgsz": 128, "batch": 8, "workers": 0}},
        "suites": {"core": {"conditions": ["real"], "detectors": ["yolo_s"], "seeds": [0]}},
        "evaluation": {"n_boot": 10, "box_min_area": 9, "box_merge_frac": 0.01},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(exp))
    assert main(["--config", str(path), "prepare"]) == 0
    return P.load_config(str(path))


def test_interrupted_yolo_run_resumes_and_only_a_finished_run_counts(cfg, monkeypatch):
    from ultralytics.engine.trainer import BaseTrainer

    from synthinspect.detect.yolo import yolo_checkpoint_finished

    rd = P.run_dir(cfg, "widget", "real", "yolo_s", 0)
    save = BaseTrainer.save_model

    def crash_after_first_epoch(self):
        save(self)
        if self.epoch == 0:
            raise SimulatedCrash("power cut")

    monkeypatch.setattr(BaseTrainer, "save_model", crash_after_first_epoch)
    with pytest.raises(SimulatedCrash):
        P.train_one(cfg, "widget", "real", "yolo_s", 0, device="cpu")
    last = rd / "weights" / "last.pt"
    assert last.exists() and not yolo_checkpoint_finished(last)
    assert not (rd / P.DONE).exists() and P.weights_of(cfg, "widget", "real", "yolo_s", 0) is None

    monkeypatch.setattr(BaseTrainer, "save_model", save)
    w = P.train_one(cfg, "widget", "real", "yolo_s", 0, device="cpu")
    assert w == last and yolo_checkpoint_finished(last)
    done = P.read_json(rd / P.DONE)
    assert done["resumed"] is True and done["weights"] == "last.pt"
    assert P.weights_of(cfg, "widget", "real", "yolo_s", 0) == last

    import synthinspect.detect.yolo as Y

    def must_not_train(*a, **k):
        raise AssertionError("a finished run was trained again")

    monkeypatch.setattr(Y, "train_yolo", must_not_train)
    assert P.train_one(cfg, "widget", "real", "yolo_s", 0, device="cpu") == last


def test_finished_checkpoint_without_marker_is_marked_not_retrained(cfg, monkeypatch):
    rd = P.run_dir(cfg, "widget", "real", "yolo_s", 0)
    w = P.train_one(cfg, "widget", "real", "yolo_s", 0, device="cpu")
    (rd / P.DONE).unlink()  # e.g. the process died between the last epoch and writing the marker

    import synthinspect.detect.yolo as Y

    monkeypatch.setattr(Y, "train_yolo", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrained")))
    assert P.train_one(cfg, "widget", "real", "yolo_s", 0, device="cpu") == w
    assert P.read_json(rd / P.DONE)["resumed"] is False
