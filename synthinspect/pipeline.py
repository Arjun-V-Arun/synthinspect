"""The experiment, stage by stage. Every stage is resumable: it skips work whose output already exists.

    prepare    -> runs/<cat>/split.json
    generate   -> runs/<cat>/gen/<method>/{images,masks,meta.jsonl}
    datasets   -> runs/<cat>/datasets/<condition>/data.yaml
    train      -> runs/<cat>/train/<condition>/<detector>/s<seed>/
    evaluate   -> runs/<cat>/preds/... and runs/<cat>/results/...
    compare    -> runs/results/comparisons.json
    explain    -> runs/<cat>/xai/...
    export     -> runs/<cat>/export/<condition>/<detector>/s<seed>/
    report     -> runs/results/RESULTS.md
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from .data.maskbank import MaskBank, PlacementPolicy
from .data.mvtec import Split, gt_for_sample, make_split
from .data.yolo_format import write_condition
from .detect.base import Detections, predict_all
from .eval import metrics as M
from .eval.review import cross_fit, fit_band, outcome
from .io import read_json, read_rgb, write_json

REPO = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- config and paths


def load_config(path: str | Path = "configs/experiment.yaml", generators: str | Path | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    gen_path = Path(generators) if generators else path.parent / "generators.yaml"
    cfg["generators"] = yaml.safe_load(gen_path.read_text(encoding="utf-8")) if gen_path.exists() else {}
    base = path.resolve().parent.parent
    for key in ("mvtec_root", "runs_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else (base / p))
    return cfg


def runs(cfg: dict) -> Path:
    return Path(cfg["runs_dir"])


def cat_dir(cfg: dict, cat: str) -> Path:
    return runs(cfg) / cat


def run_dir(cfg: dict, cat: str, cond: str, det: str, seed: int) -> Path:
    return cat_dir(cfg, cat) / "train" / cond / det / f"s{seed}"


def load_split(cfg: dict, cat: str) -> Split:
    return Split.load(cat_dir(cfg, cat) / "split.json")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _categories(cfg: dict, only: Iterable[str] | None) -> list[str]:
    cats = list(cfg["categories"])
    return [c for c in cats if not only or c in only]


# ----------------------------------------------------------------------------- prepare / generate / datasets


def prepare(cfg: dict, categories: Iterable[str] | None = None) -> dict:
    summary = {}
    for cat in _categories(cfg, categories):
        sp = make_split(cfg["mvtec_root"], cat, cfg["split"]["train_fraction_den"])
        sp.save(cat_dir(cfg, cat) / "split.json")
        per_type = {t: sum(s.defect_type == t for s in sp.defect_train) for t in sp.defect_types}
        summary[cat] = {
            "normal_train": len(sp.normal_train),
            "defect_train": per_type,
            "test_defect": sum(s.is_defect for s in sp.test),
            "test_good": sum(not s.is_defect for s in sp.test),
        }
        _log(f"{cat}: {summary[cat]}")
    return summary


def mask_bank(cfg: dict, cat: str, sp: Split) -> MaskBank:
    pl = dict(cfg["categories"][cat].get("placement", {}))
    return MaskBank(sp.defect_train, PlacementPolicy(**pl))


def synthesis_split(cfg: dict, cat: str, sp: Split) -> Split:
    """The split generators see: defect types listed under ``skip_synthesis`` are removed.

    Whole-part defects (metal_nut "flip": the entire nut upside down) cannot be made
    by painting into a region of a good part; copy-paste stacked one nut on another.
    Every generator skips them, so the comparison stays fair, and the detectors still
    train on the real examples (datasets are built from the full split).
    """
    skip = set(cfg["categories"][cat].get("skip_synthesis", []))
    unknown = skip - set(sp.defect_types)
    if unknown:
        raise ValueError(f"skip_synthesis lists types not in {cat}: {sorted(unknown)}")
    if not skip:
        return sp
    out = Split.from_dict(sp.to_dict())
    out.defect_types = [t for t in sp.defect_types if t not in skip]
    out.defect_train = [s for s in sp.defect_train if s.defect_type not in skip]
    return out


def generate(cfg: dict, cat: str, method: str, n_per_type: int | None = None) -> Path:
    from .gen import build_generator
    from .gen.base import generate as run_generation

    sp = synthesis_split(cfg, cat, load_split(cfg, cat))
    bank = mask_bank(cfg, cat, sp)
    gcfg = copy.deepcopy(cfg["generators"].get(method, {}))
    gen = build_generator(method, gcfg, cat)
    out = cat_dir(cfg, cat) / "gen" / method
    _log(f"{cat}/{method}: fitting")
    gen.fit(sp, bank, out / "model")
    n = n_per_type or cfg["generation"]["n_per_type"]
    _log(f"{cat}/{method}: generating {n} per defect type")
    fg_mode = cfg["categories"][cat].get("placement", {}).get("foreground", "auto")
    rows = run_generation(gen, sp, bank, out, n, seed=cfg["generation"]["seed"], foreground_mode=fg_mode)
    _log(f"{cat}/{method}: wrote {len(rows)} new images")
    return out


def build_datasets(cfg: dict, cat: str, conditions: Iterable[str]) -> dict[str, Path]:
    sp = load_split(cfg, cat)
    ds = cfg["dataset"]
    ev = cfg["evaluation"]
    out = {}
    for cond in conditions:
        gen_dirs = [] if cond == "real" else [cat_dir(cfg, cat) / "gen" / cond]
        for g in gen_dirs:
            if not (g / "meta.jsonl").exists():
                raise FileNotFoundError(f"Generate {cond} for {cat} first (missing {g})")
        target = cat_dir(cfg, cat) / "datasets" / cond / "data.yaml"
        if target.exists():
            out[cond] = target
            continue
        out[cond] = write_condition(
            sp,
            target.parent,
            gen_dirs,
            n_generated_per_type=cfg["generation"]["n_per_type"],
            n_normals=ds.get("n_normals"),
            class_mode=ds["class_mode"],
            min_area=ev["box_min_area"],
            merge_frac=ev["box_merge_frac"],
        )
        _log(f"{cat}: dataset {cond} -> {yaml.safe_load((target.parent / 'counts.yaml').read_text())}")
    return out


# ----------------------------------------------------------------------------- training


def resolve_conditions(cfg: dict, cat: str, conditions: list[str]) -> list[str]:
    out = []
    for c in conditions:
        if c == "best":
            out.append(best_generator(cfg, cat))
        else:
            out.append(c)
    return list(dict.fromkeys(out))


def best_generator(cfg: dict, cat: str, metric: str = "image_ap") -> str:
    """The generator with the highest mean metric over seeds in the core suite (never 'real')."""
    suite = cfg["suites"]["core"]
    scores = {}
    for cond in suite["conditions"]:
        if cond == "real":
            continue
        vals = []
        for det in suite["detectors"]:
            for seed in suite["seeds"]:
                p = cat_dir(cfg, cat) / "results" / cond / det / f"s{seed}.json"
                if p.exists():
                    vals.append(read_json(p)["metrics"][metric]["value"])
        if vals:
            scores[cond] = float(np.mean(vals))
    if not scores:
        raise RuntimeError(f"No core results for {cat}; run the core suite first")
    return max(scores, key=scores.get)


def train_one(cfg: dict, cat: str, cond: str, det_name: str, seed: int, device: str | None = None) -> Path:
    d = cfg["detectors"][det_name]
    rd = run_dir(cfg, cat, cond, det_name, seed)
    data = build_datasets(cfg, cat, [cond])[cond]
    if d["family"] == "yolo":
        last = rd / "weights" / "last.pt"
        if last.exists():
            return last
        from .detect.yolo import train_yolo

        aug = cfg["categories"][cat].get("yolo_aug", {})
        return train_yolo(
            data, rd, model=d["model"], pretrained=d["pretrained"], epochs=d["epochs"], imgsz=d["imgsz"],
            batch=d["batch"], seed=seed, workers=d.get("workers", 2), device=device, aug=aug,
        )
    if d["family"] == "rfdetr":
        for name in ("last_ema.pth", "last.pth"):
            if (rd / name).exists():
                return rd / name
        from .detect.rfdetr_wrap import train_rfdetr

        return train_rfdetr(
            data.parent, rd, size=d["size"], pretrained=d["pretrained"], epochs=d["epochs"], batch=d["batch"],
            grad_accum=d.get("grad_accum", 1), seed=seed, workers=d.get("workers", 2), device=device,
        )
    raise ValueError(d["family"])


def weights_of(cfg: dict, cat: str, cond: str, det_name: str, seed: int) -> Path | None:
    rd = run_dir(cfg, cat, cond, det_name, seed)
    for p in (rd / "weights" / "last.pt", rd / "last_ema.pth", rd / "last.pth"):
        if p.exists():
            return p
    return None


def suite_runs(cfg: dict, suite: str, categories=None, conditions=None, detectors=None, seeds=None):
    s = cfg["suites"][suite]
    for cat in _categories(cfg, categories):
        conds = resolve_conditions(cfg, cat, s["conditions"])
        for cond in conds:
            if conditions and cond not in conditions:
                continue
            for det in s["detectors"]:
                if detectors and det not in detectors:
                    continue
                for seed in s["seeds"]:
                    if seeds is not None and seed not in seeds:
                        continue
                    yield cat, cond, det, seed


def train(cfg: dict, suite: str, device: str | None = None, **filters) -> None:
    for cat, cond, det, seed in suite_runs(cfg, suite, **filters):
        _log(f"train {cat}/{cond}/{det}/s{seed}")
        train_one(cfg, cat, cond, det, seed, device)


# ----------------------------------------------------------------------------- evaluation


def test_set(cfg: dict, sp: Split) -> tuple[list[str], np.ndarray, list[str], list[np.ndarray]]:
    ev = cfg["evaluation"]
    paths, labels, types, gts = [], [], [], []
    for s in sp.test:
        boxes, mask = gt_for_sample(s, min_area=ev["box_min_area"], merge_dist=0)
        if mask is not None:
            from .data.mvtec import boxes_from_mask

            boxes = boxes_from_mask(mask, ev["box_min_area"], max(1, int(ev["box_merge_frac"] * max(mask.shape))))
        paths.append(s.image)
        labels.append(int(s.is_defect))
        types.append(s.defect_type)
        gts.append(boxes)
    return paths, np.array(labels), types, gts


def save_preds(path: Path, images: list[str], preds: list[Detections]) -> None:
    write_json(path, [{"image": im, **p.to_dict()} for im, p in zip(images, preds)])


def load_preds(path: Path) -> list[Detections]:
    rows = read_json(path)
    return [
        Detections(np.array(r["boxes"], np.float32).reshape(-1, 4), np.array(r["scores"], np.float32), np.array(r["classes"], np.int64))
        for r in rows
    ]


def predict_test(cfg: dict, cat: str, detector, cache: Path) -> tuple[EvalData, list[str]]:
    sp = load_split(cfg, cat)
    paths, labels, types, gts = test_set(cfg, sp)
    if cache.exists():
        preds = load_preds(cache)
    else:
        preds = []
        for i in range(0, len(paths), 8):
            preds.extend(predict_all(detector, [read_rgb(p) for p in paths[i : i + 8]], batch=8))
        save_preds(cache, paths, preds)
    return M.EvalData(labels, types, preds, gts), paths


EvalData = M.EvalData


def score_run(cfg: dict, data: M.EvalData) -> dict:
    r = cfg["review"]
    res = {"metrics": M.with_ci(data, n_boot=cfg["evaluation"]["n_boot"])}
    res["review"] = cross_fit(
        data.scores, data.labels, r["max_escape"], r["max_false_reject"], r["repeats"], strata=data.types
    )
    # Per defect type: AP of that type's images against all good images.
    per_type = {}
    good = data.labels == 0
    for t in sorted(set(data.types) - {"good"}):
        idx = np.where(good | (np.array(data.types) == t))[0]
        per_type[t] = M.compute(data, idx)
    res["per_type"] = per_type
    return res


def evaluate(cfg: dict, suite: str, **filters) -> None:
    from .detect.runtime import load_detector

    for cat, cond, det, seed in suite_runs(cfg, suite, **filters):
        out = cat_dir(cfg, cat) / "results" / cond / det / f"s{seed}.json"
        if out.exists():
            continue
        w = weights_of(cfg, cat, cond, det, seed)
        if w is None:
            _log(f"skip {cat}/{cond}/{det}/s{seed}: not trained")
            continue
        _log(f"evaluate {cat}/{cond}/{det}/s{seed}")
        dcfg = cfg["detectors"][det]
        detector = None
        cache = cat_dir(cfg, cat) / "preds" / cond / det / f"s{seed}.json"
        if not cache.exists():
            detector = load_detector(w, dcfg.get("imgsz", 640))
        data, _ = predict_test(cfg, cat, detector, cache)
        res = score_run(cfg, data)
        res.update({"category": cat, "condition": cond, "detector": det, "seed": seed, "weights": str(w)})
        write_json(out, res)


def compare(cfg: dict, suite: str = "core", references: tuple[str, ...] = ("rgcb", "real"), metrics=("image_ap", "image_auroc", "box_ap10")) -> dict:
    """Paired bootstrap differences on the same test images, pooled over seeds."""
    s = cfg["suites"][suite]
    n_boot = max(200, cfg["evaluation"]["n_boot"] // max(1, len(s["seeds"])))
    out: dict = {}
    for cat in _categories(cfg, None):
        sp = load_split(cfg, cat)
        _, labels, types, gts = test_set(cfg, sp)
        for det in s["detectors"]:
            conds = resolve_conditions(cfg, cat, s["conditions"])
            for ref in references:
                if ref not in conds:
                    continue
                for cond in conds:
                    if cond == ref:
                        continue
                    for metric in metrics:
                        draws, points = [], []
                        for seed in s["seeds"]:
                            pa = cat_dir(cfg, cat) / "preds" / cond / det / f"s{seed}.json"
                            pb = cat_dir(cfg, cat) / "preds" / ref / det / f"s{seed}.json"
                            if not (pa.exists() and pb.exists()):
                                continue
                            a = M.EvalData(labels, types, load_preds(pa), gts)
                            b = M.EvalData(labels, types, load_preds(pb), gts)
                            points.append(M.compute(a)[metric] - M.compute(b)[metric])
                            for idx in M.stratified_bootstrap(labels, n_boot, seed):
                                draws.append(M.compute(a, idx)[metric] - M.compute(b, idx)[metric])
                        if not points:
                            continue
                        d = np.array(draws, float)
                        d = d[~np.isnan(d)]
                        key = f"{cat}|{det}|{cond}-vs-{ref}|{metric}"
                        out[key] = {
                            "diff": float(np.mean(points)),
                            "ci_low": float(np.percentile(d, 2.5)),
                            "ci_high": float(np.percentile(d, 97.5)),
                            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                            "n_seeds": len(points),
                        }
    write_json(runs(cfg) / "results" / "comparisons.json", out)
    return out


# ----------------------------------------------------------------------------- explanations


def explain(cfg: dict, cat: str, detector: str = "yolo26n_pre", seed: int = 0, conditions: list[str] | None = None, runtime_check: bool = True) -> dict:
    """ODAM localisation per condition, faithfulness, explanation gate and (if exported) FP32-vs-INT8 agreement."""
    from .detect.yolo import TorchYOLO
    from .xai import study

    x = cfg["xai"]
    sp = load_split(cfg, cat)
    paths, labels, types, gts = test_set(cfg, sp)
    conds = conditions or resolve_conditions(cfg, cat, cfg["suites"]["core"]["conditions"])
    out_dir = cat_dir(cfg, cat) / "xai" / detector / f"s{seed}"
    result: dict = {"localisation": {}, "gate": {}}
    for cond in conds:
        w = weights_of(cfg, cat, cond, detector, seed)
        if w is None or cfg["detectors"][detector]["family"] != "yolo":
            continue
        cache = out_dir / f"odam_{cond}.json"
        if cache.exists():
            recs = read_json(cache)
        else:
            _log(f"ODAM {cat}/{cond}")
            det = TorchYOLO(w, cfg["detectors"][detector]["imgsz"])
            recs = study.odam_records(det, sp.test, gts, x["pointing_tolerance"])
            write_json(cache, recs)
        result["localisation"][cond] = study.summarise_localisation(recs)
        result["gate"][cond] = study.gate_analysis(recs, cfg["review"])

    best = best_generator(cfg, cat)
    w = weights_of(cfg, cat, best, detector, seed)
    if w is not None:
        det = TorchYOLO(w, cfg["detectors"][detector]["imgsz"])
        defects = [s for s in sp.test if s.is_defect]
        fcache = out_dir / f"faithfulness_{best}.json"
        if not fcache.exists():
            _log(f"faithfulness {cat}/{best}")
            write_json(fcache, study.faithfulness(det, defects, x["faithfulness_images"], x["drise"]))
        result["faithfulness"] = {"condition": best, **read_json(fcache)}
        exp = cat_dir(cfg, cat) / "export" / best / detector / f"s{seed}"
        if runtime_check and (exp / "openvino" / "int8" / "model.xml").exists():
            qcache = out_dir / f"quantisation_{best}.json"
            if not qcache.exists():
                from .detect.runtime import OnnxDetector, OpenVINODetector

                _log(f"FP32 vs INT8 explanations {cat}/{best}")
                imgsz = cfg["detectors"][detector]["imgsz"]
                q = study.quantisation_agreement(
                    OnnxDetector(exp / "model.onnx", imgsz), OpenVINODetector(exp / "openvino" / "int8" / "model.xml", imgsz),
                    sp.test, x["drise_images"], x["drise"], x["pointing_tolerance"],
                )
                write_json(qcache, q)
            result["quantisation"] = {"condition": best, **read_json(qcache)}
    write_json(out_dir / "summary.json", result)
    return result


# ----------------------------------------------------------------------------- ship small


def export(cfg: dict, cat: str, cond: str | None = None, detector: str = "yolo26n_pre", seed: int = 0) -> dict:
    from .deploy.bench import export_and_bench

    cond = cond or best_generator(cfg, cat)
    w = weights_of(cfg, cat, cond, detector, seed)
    if w is None:
        raise FileNotFoundError(f"Train {cat}/{cond}/{detector}/s{seed} first")
    sp = load_split(cfg, cat)
    paths, labels, types, gts = test_set(cfg, sp)
    calib = [s.image for s in sp.normal_train] + [s.image for s in sp.defect_train]
    gen_dir = cat_dir(cfg, cat) / "gen" / cond / "images"
    if gen_dir.exists():
        calib += [str(p) for p in sorted(gen_dir.glob("*.png"))]
    dcfg = cfg["detectors"][detector]
    out = cat_dir(cfg, cat) / "export" / cond / detector / f"s{seed}"
    data = M.EvalData(labels, types, [Detections()] * len(paths), gts)
    return export_and_bench(w, dcfg["family"], dcfg.get("imgsz", 640), out, calib, paths, data, cfg["export"], log=_log)


# ----------------------------------------------------------------------------- report


def report(cfg: dict) -> Path:
    from .report import build_report

    path = build_report(cfg)
    _log(f"wrote {path}")
    return path
