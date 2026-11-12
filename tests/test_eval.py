import numpy as np
import pytest

from synthinspect.detect.base import Detections
from synthinspect.eval.metrics import EvalData, auroc, average_precision, box_ap, iou_matrix, paired_difference, with_ci
from synthinspect.eval.review import Band, cross_fit, fit_band, outcome


def test_auroc_matches_sklearn_cases():
    assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # ties count half
    assert auroc([1, 1, 0], [0.9, 0.8, 0.1]) == 1.0


def test_average_precision_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 40)
        y[0] = 1
        s = np.round(rng.random(40), 1)  # forces ties
        assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert iou_matrix(a, b)[0].tolist() == pytest.approx([50 / 150, 0])


def _det(boxes, scores):
    return Detections(np.array(boxes, np.float32).reshape(-1, 4), np.array(scores, np.float32), np.zeros(len(scores), np.int64))


def test_box_ap_perfect_and_duplicates():
    gts = [np.array([[0, 0, 10, 10]], float), np.zeros((0, 4))]
    perfect = [_det([[0, 0, 10, 10]], [0.9]), _det([], [])]
    assert box_ap(perfect, gts) == pytest.approx(1.0)
    # a duplicate detection of the same defect is a false positive
    dup = [_det([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8]), _det([[50, 50, 60, 60]], [0.95])]
    assert box_ap(dup, gts) == pytest.approx(0.5)  # precision 1/2 when recall reaches 1


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    labels = np.array([1] * 20 + [0] * 20)
    preds = [_det([[0, 0, 5, 5]], [rng.uniform(0.4, 1.0) if l else rng.uniform(0.0, 0.6)]) for l in labels]
    gts = [np.array([[0, 0, 5, 5]], float) if l else np.zeros((0, 4)) for l in labels]
    d = EvalData(labels, ["x"] * 40, preds, gts)
    res = with_ci(d, n_boot=200)
    for v in res.values():
        assert v["ci_low"] <= v["value"] <= v["ci_high"]
    same = paired_difference(d, d, "image_auroc", n_boot=100)
    assert same["diff"] == 0 and same["p_two_sided"] == 1.0


def test_band_respects_targets_on_calibration_data():
    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 200)
    s = np.where(y == 1, rng.normal(0.7, 0.15, 300), rng.normal(0.3, 0.15, 300))
    band = fit_band(s, y, max_escape=0.0, max_false_reject=0.01)
    out = outcome(band, s, y)
    assert out["n_escapes"] == 0
    assert out["false_reject_rate"] <= 0.01
    assert 0 < out["automation"] < 1


def test_band_never_overlaps():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.95, 0.2])  # hopeless model
    band = fit_band(s, y, 0.0, 0.0)
    assert band.t_pass <= band.t_reject
    assert outcome(band, s, y)["n_escapes"] == 0


def test_cross_fit_on_separable_scores_automates_everything():
    y = np.array([1] * 40 + [0] * 40)
    s = np.where(y == 1, 0.9, 0.1) + np.linspace(0, 0.01, 80)
    res = cross_fit(s, y, repeats=5)
    assert res["automation"] == pytest.approx(1.0)
    assert res["escape_rate"] == 0 and res["false_reject_rate"] == 0


def test_band_decide():
    d = Band(0.2, 0.8).decide(np.array([0.1, 0.2, 0.5, 0.8, 0.9]))
    assert d.tolist() == [0, 1, 1, 1, 2]
