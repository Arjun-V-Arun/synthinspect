"""The explainability experiments.

1. Localisation by training data: for each generator condition, explain the
   top detection on every defective test part with ODAM and score it against
   the true defect mask. Question: does synthetic training data make the model
   look somewhere other than the defect (for example at paste seams)?
2. Faithfulness: deletion/insertion for ODAM vs D-RISE on the same detections.
3. Quantisation: D-RISE on the FP32 export and on the INT8 model for the same
   detection. Question: did INT8 change what the model relies on?
4. Explanation gate: does "share of the explanation inside the detection's own
   box" separate true from false top detections, and does sending low-share
   rejections to review help?
"""

from __future__ import annotations

import numpy as np

from ..eval.metrics import auroc, iou_matrix
from ..eval.review import cross_fit
from ..io import read_mask, read_rgb, write_json
from .drise import drise
from .metrics import box_region, deletion_insertion, energy_in, pointing_game, saliency_agreement
from .odam import odam


def _boot_mean(x: list[float], n: int = 1000, seed: int = 0) -> dict[str, float]:
    a = np.asarray(x, float)
    if len(a) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return {"mean": float(a.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)), "n": int(len(a))}


def odam_records(det, samples, gts, tolerance: int = 15, iou_tp: float = 0.1) -> list[dict]:
    """One record per test part: top detection, whether it hits the defect, and ODAM scores."""
    recs = []
    for s, gt in zip(samples, gts):
        img = read_rgb(s.image)
        maps, raw = odam(det, img, indices=[0])
        top = det.predict([img], min_score=0.0)[0].top(1)
        if len(top) == 0:
            continue
        box = top.boxes[0]
        sal = maps[0]
        is_tp = bool(s.is_defect and len(gt) and iou_matrix(box[None], gt).max() >= iou_tp)
        rec = {
            "image": s.image,
            "defect_type": s.defect_type,
            "label": int(s.is_defect),
            "score": float(top.scores[0]),
            "tp": is_tp,
            "inbox_energy": energy_in(sal, box_region(box, sal.shape)),
        }
        if s.is_defect:
            m = read_mask(s.mask)
            rec["pointing_game"] = pointing_game(sal, m, tolerance)
            rec["mask_energy"] = energy_in(sal, m)
        recs.append(rec)
    return recs


def summarise_localisation(recs: list[dict]) -> dict:
    tp = [r for r in recs if r["tp"]]
    return {
        "n_tp": len(tp),
        "pointing_game": _boot_mean([float(r["pointing_game"]) for r in tp]),
        "mask_energy": _boot_mean([r["mask_energy"] for r in tp]),
    }


def gate_analysis(recs: list[dict], review_cfg: dict, taus=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)) -> dict:
    """Does in-box explanation energy tell true detections from false ones, and does gating rejects help?"""
    e = np.array([r["inbox_energy"] for r in recs])
    tp = np.array([r["tp"] for r in recs])
    # False top detections: on good parts, or on defective parts but somewhere else.
    out = {"auroc_tp_vs_fp": auroc(tp.astype(int), e), "n_tp": int(tp.sum()), "n_fp": int((~tp).sum())}
    scores = np.array([r["score"] for r in recs])
    labels = np.array([r["label"] for r in recs])
    strata = [r["defect_type"] for r in recs]
    rows = []
    for tau in taus:
        res = cross_fit(
            scores, labels, review_cfg["max_escape"], review_cfg["max_false_reject"], review_cfg["repeats"], strata=strata, gate=e, gate_tau=tau
        )
        rows.append({"tau": tau, **res})
    out["by_tau"] = rows
    return out


def faithfulness(det, samples, n: int, drise_cfg: dict, seed: int = 0) -> dict:
    rows = {"odam": [], "drise": []}
    for s in samples[:n]:
        img = read_rgb(s.image)
        top = det.predict([img], min_score=0.0)[0].top(1)
        if len(top) == 0:
            continue
        box = top.boxes[0]
        maps, _ = odam(det, img, indices=[0])
        rows["odam"].append(deletion_insertion(det, img, box, maps[0]))
        rows["drise"].append(deletion_insertion(det, img, box, drise(det, img, box, seed=seed, **drise_cfg)))
    return {
        k: {"deletion": _boot_mean([r["deletion"] for r in v if np.isfinite(r["deletion"])]), "insertion": _boot_mean([r["insertion"] for r in v if np.isfinite(r["insertion"])])}
        for k, v in rows.items()
    }


def quantisation_agreement(det_fp32, det_int8, samples, n: int, drise_cfg: dict, tolerance: int = 15, seed: int = 0) -> dict:
    """Explain the same FP32 detection with D-RISE on both runtimes and compare the maps."""
    rows = []
    for s in [x for x in samples if x.is_defect][:n]:
        img = read_rgb(s.image)
        top = det_fp32.predict([img], min_score=0.0)[0].top(1)
        if len(top) == 0:
            continue
        box = top.boxes[0]
        a = drise(det_fp32, img, box, seed=seed, **drise_cfg)
        b = drise(det_int8, img, box, seed=seed, **drise_cfg)  # same masks: differences come from the model only
        m = read_mask(s.mask)
        rows.append({
            **saliency_agreement(a, b),
            "pg_fp32": float(pointing_game(a, m, tolerance)),
            "pg_int8": float(pointing_game(b, m, tolerance)),
            "energy_fp32": energy_in(a, m),
            "energy_int8": energy_in(b, m),
        })
    return {k: _boot_mean([r[k] for r in rows]) for k in (rows[0] if rows else {})}


def save(path, obj) -> None:
    write_json(path, obj)
