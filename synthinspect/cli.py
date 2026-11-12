"""Command line: python -m synthinspect <stage> [options]. Run with --help for each stage."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("YOLO_OFFLINE", "1")


def main(argv: list[str] | None = None) -> int:
    from . import pipeline as P

    ap = argparse.ArgumentParser(prog="synthinspect", description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build the few-shot splits")
    p.add_argument("--category", nargs="*")

    p = sub.add_parser("generate", help="fit a generator and write synthetic defects")
    p.add_argument("--category", required=True)
    p.add_argument("--method", required=True, choices=["copypaste", "rgcb", "defectfill", "magic"])
    p.add_argument("--n", type=int, help="images per defect type (default from config)")

    for name in ("train", "evaluate"):
        p = sub.add_parser(name, help=f"{name} every run in a suite (skips finished runs)")
        p.add_argument("--suite", default="core")
        p.add_argument("--category", nargs="*")
        p.add_argument("--condition", nargs="*")
        p.add_argument("--detector", nargs="*")
        p.add_argument("--seed", nargs="*", type=int)
        if name == "train":
            p.add_argument("--device", help="e.g. 0 for the first GPU, cpu")

    p = sub.add_parser("compare", help="paired bootstrap differences between conditions")
    p.add_argument("--suite", default="core")

    p = sub.add_parser("explain", help="explanation studies for one category")
    p.add_argument("--category", required=True)
    p.add_argument("--detector", default="yolo26n_pre")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("export", help="ONNX / OpenVINO FP16 / INT8 export, accuracy and latency")
    p.add_argument("--category", required=True)
    p.add_argument("--condition", help="default: the best generator")
    p.add_argument("--detector", default="yolo26n_pre")
    p.add_argument("--seed", type=int, default=0)

    sub.add_parser("report", help="write runs/results/RESULTS.md and the README headline")

    p = sub.add_parser("demo", help="package a model and samples for the browser demo (web/)")
    p.add_argument("--category", required=True)
    p.add_argument("--condition")
    p.add_argument("--detector", default="yolo26n_pre")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=8)

    p = sub.add_parser("serve", help="run the inspection API locally")
    p.add_argument("--model-dir", required=True, help="a folder made by `demo` or containing model.xml/model.onnx + config.json")
    p.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        os.environ["SYNTHINSPECT_MODEL_DIR"] = a.model_dir
        import uvicorn

        uvicorn.run("synthinspect.serve.app:app", host="0.0.0.0", port=a.port)
        return 0

    cfg = P.load_config(a.config)
    if a.cmd == "prepare":
        P.prepare(cfg, a.category)
    elif a.cmd == "generate":
        P.generate(cfg, a.category, a.method, a.n)
    elif a.cmd in ("train", "evaluate"):
        filters = dict(categories=a.category, conditions=a.condition, detectors=a.detector, seeds=a.seed)
        if a.cmd == "train":
            P.train(cfg, a.suite, device=a.device, **filters)
        else:
            P.evaluate(cfg, a.suite, **filters)
    elif a.cmd == "compare":
        P.compare(cfg, a.suite)
    elif a.cmd == "explain":
        P.explain(cfg, a.category, a.detector, a.seed)
    elif a.cmd == "export":
        P.export(cfg, a.category, a.condition, a.detector, a.seed)
    elif a.cmd == "report":
        P.report(cfg)
    elif a.cmd == "demo":
        from .serve.package import package_demo

        package_demo(cfg, a.category, a.condition, a.detector, a.seed, a.n_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
