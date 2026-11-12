"""Turn the result files into RESULTS.md, and a short headline table in the README."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .io import read_json

LABELS = {
    "real": "Real only",
    "copypaste": "Copy-paste (no NST)",
    "rgcb": "RGCB (copy-paste + NST)",
    "defectfill": "DefectFill (re-impl.)",
    "magic": "MAGIC (re-impl., LoRA)",
}


def _fmt(vals: list[float], digits: int = 3) -> str:
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals):.{digits}f}"


def _pct(v: float) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{100 * v:.1f}%"


def _results(cfg: dict, cat: str, suite: str) -> dict[tuple[str, str], list[dict]]:
    from .pipeline import cat_dir, resolve_conditions

    s = cfg["suites"][suite]
    out: dict[tuple[str, str], list[dict]] = {}
    try:
        conds = resolve_conditions(cfg, cat, s["conditions"])
    except RuntimeError:
        conds = [c for c in s["conditions"] if c != "best"]
    for cond in conds:
        for det in s["detectors"]:
            rows = []
            for seed in s["seeds"]:
                p = cat_dir(cfg, cat) / "results" / cond / det / f"s{seed}.json"
                if p.exists():
                    rows.append(read_json(p))
            if rows:
                out[(cond, det)] = rows
    return out


def _table(rows: dict[tuple[str, str], list[dict]], by: str) -> list[str]:
    head = "| {} | Image AP | Image AUROC | Box AP@0.1 | Box AP@0.5 | Automated | Escaped defects | False rejects | Seeds |".format(
        "Training data" if by == "cond" else "Detector"
    )
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for (cond, det), rs in rows.items():
        name = LABELS.get(cond, cond) if by == "cond" else f"{det} ({LABELS.get(cond, cond)})"
        m = lambda k: _fmt([r["metrics"][k]["value"] for r in rs])  # noqa: E731
        rv = lambda k: _pct(float(np.mean([r["review"][k] for r in rs])))  # noqa: E731
        lines.append(
            f"| {name} | {m('image_ap')} | {m('image_auroc')} | {m('box_ap10')} | {m('box_ap50')} | {rv('automation')} | {rv('escape_rate')} | {rv('false_reject_rate')} | {len(rs)} |"
        )
    return lines


def build_report(cfg: dict) -> Path:
    from .pipeline import _categories, cat_dir, runs

    out = runs(cfg) / "results" / "RESULTS.md"
    comps_path = runs(cfg) / "results" / "comparisons.json"
    comps = read_json(comps_path) if comps_path.exists() else {}
    L = ["# Results", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Values are mean ± sd over seeds; review-band rates are out-of-fold (two-fold cross-fitting).", ""]
    headline = []
    for cat in _categories(cfg, None):
        L += [f"## {cat.replace('_', ' ').title()}", ""]
        core = _results(cfg, cat, "core")
        if core:
            L += ["### Generator Comparison", ""] + _table(core, "cond") + [""]
            for (cond, det), rs in core.items():
                headline.append((cat, cond, float(np.mean([r["metrics"]["image_ap"]["value"] for r in rs])), float(np.mean([r["review"]["automation"] for r in rs]))))
        diffs = {k: v for k, v in comps.items() if k.startswith(f"{cat}|")}
        if diffs:
            L += ["### Paired Differences (Same Test Images, Pooled Over Seeds)", "", "| Comparison | Metric | Difference | 95% CI | p |", "|---|---|---|---|---|"]
            for k, v in diffs.items():
                _, det, pair, metric = k.split("|")
                a, b = pair.split("-vs-")
                L.append(f"| {LABELS.get(a, a)} vs {LABELS.get(b, b)} | {metric} | {v['diff']:+.3f} | [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] | {v['p_two_sided']:.3f} |")
            L.append("")
        models = _results(cfg, cat, "models")
        if models:
            L += ["### Detectors: Pretrained vs From Scratch vs Transformer", ""] + _table(models, "det") + [""]

        xs = sorted((cat_dir(cfg, cat) / "xai").glob("*/s*/summary.json"))
        for xp in xs[:1]:
            x = read_json(xp)
            if x.get("localisation"):
                L += ["### Where the Model Looks (ODAM on True Detections)", "", "| Training data | True detections | Pointing game | Energy on defect |", "|---|---|---|---|"]
                for cond, v in x["localisation"].items():
                    pg, me = v["pointing_game"], v["mask_energy"]
                    L.append(f"| {LABELS.get(cond, cond)} | {v['n_tp']} | {_pct(pg['mean'])} [{_pct(pg['ci_low'])}, {_pct(pg['ci_high'])}] | {_pct(me['mean'])} [{_pct(me['ci_low'])}, {_pct(me['ci_high'])}] |")
                L.append("")
            if x.get("gate"):
                L += ["### Explanation Gate", "", "Share of the explanation inside the detection's own box, used to send would-be rejections to review.", ""]
                for cond, g in x["gate"].items():
                    L.append(f"- {LABELS.get(cond, cond)}: AUROC for true vs false top detections = {g['auroc_tp_vs_fp']:.3f} ({g['n_tp']} true, {g['n_fp']} false)")
                best = next(iter(x["gate"].values()))
                L += ["", "| Gate threshold | Automated | Escaped defects | False rejects |", "|---|---|---|---|"]
                for r in best["by_tau"]:
                    L.append(f"| {r['tau']:.1f} | {_pct(r['automation'])} | {_pct(r['escape_rate'])} | {_pct(r['false_reject_rate'])} |")
                L.append("")
            if x.get("faithfulness"):
                f = x["faithfulness"]
                L += [f"### Faithfulness ({LABELS.get(f['condition'], f['condition'])})", "", "| Method | Deletion (lower is better) | Insertion (higher is better) |", "|---|---|---|"]
                for meth in ("odam", "drise"):
                    L.append(f"| {meth.upper().replace('DRISE', 'D-RISE')} | {f[meth]['deletion']['mean']:.3f} | {f[meth]['insertion']['mean']:.3f} |")
                L.append("")
            if x.get("quantisation"):
                q = x["quantisation"]
                L += [
                    "### Did INT8 Change the Explanations?",
                    "",
                    f"D-RISE with identical masks on the FP32 export and the INT8 model ({q.get('spearman', {}).get('n', 0)} defective parts):",
                    "",
                    f"- Rank correlation of the maps: {q['spearman']['mean']:.3f}",
                    f"- Overlap of the hottest 10%: {q['top10_iou']['mean']:.3f}",
                    f"- Pointing game FP32 {_pct(q['pg_fp32']['mean'])}, INT8 {_pct(q['pg_int8']['mean'])}",
                    "",
                ]
        for rp in sorted((cat_dir(cfg, cat) / "export").glob("*/*/s*/report.json"))[:1]:
            r = read_json(rp)
            L += ["### Ship Small", "", f"Measured on: {r['machine'].get('cpu', r['machine'].get('processor'))}. Image size {r['imgsz']}.", ""]
            L += ["| Runtime | Size (MB) | p50 (ms) | p95 (ms) | Image AP | Δ AP | Box AP@0.1 |", "|---|---|---|---|---|---|---|"]
            for name, v in r["runtimes"].items():
                L.append(f"| {name} | {v['size_mb']} | {v['p50_ms']} | {v['p95_ms']} | {v['image_ap']:.3f} | {v['d_image_ap']:+.3f} | {v['box_ap10']:.3f} |")
            L += ["", f"Release gate ({r['release_gate']['rule']}): {'passed' if r['release_gate']['passed'] else 'FAILED'}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    _update_readme(headline)
    return out


def _update_readme(headline: list[tuple[str, str, float, float]]) -> None:
    from .pipeline import REPO

    readme = REPO / "README.md"
    if not readme.exists() or not headline:
        return
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    if start not in text or end not in text:
        return
    rows = ["| Category | Training data | Image AP | Automated by the review band |", "|---|---|---|---|"]
    rows += [f"| {c} | {LABELS.get(k, k)} | {ap:.3f} | {100 * auto:.0f}% |" for c, k, ap, auto in headline]
    block = start + "\n" + "\n".join(rows) + "\n\nFull tables, confidence intervals and paired tests: `runs/results/RESULTS.md`.\n" + end
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    readme.write_text(before + block + after, encoding="utf-8")
