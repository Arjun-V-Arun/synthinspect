"""Export one trained run to every runtime, then measure size, latency and accuracy for each.

Accuracy is measured on the same test set as the PyTorch model, so the table
shows exactly what quantisation costs. Latency is measured on whatever machine
runs this (your laptop CPU; the README says which).
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np

from ..eval import metrics as M
from ..io import read_rgb, write_json


def latency(det, image: np.ndarray, warmup: int = 20, runs: int = 200) -> dict[str, float]:
    for _ in range(warmup):
        det.predict([image])
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        det.predict([image])
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean()), "fps": float(1000 / a.mean())}


def size_mb(path: Path) -> float:
    path = Path(path)
    files = [path] + ([path.with_suffix(".bin")] if path.suffix == ".xml" else [])
    return sum(f.stat().st_size for f in files if f.exists()) / 1e6


def machine() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor() or platform.machine(), "python": platform.python_version()}
    try:
        import openvino as ov

        info["cpu"] = ov.Core().get_property("CPU", "FULL_DEVICE_NAME")
    except Exception:
        pass
    return info


def export_and_bench(
    weights: Path,
    family: str,
    imgsz: int,
    out_dir: Path,
    calib_paths: list[str],
    test_paths: list[str],
    test_data: M.EvalData,
    export_cfg: dict,
    log=print,
) -> dict:
    from ..detect.runtime import OnnxDetector, OpenVINODetector, load_detector
    from .export import calibration_batches, onnx_fp16, onnx_int8, to_openvino

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "yolo26" if family == "yolo" else "rfdetr"
    if family == "yolo":
        from .export import export_yolo_onnx

        onnx_path = export_yolo_onnx(weights, out_dir / "model.onnx", imgsz)
    else:
        from ..detect.rfdetr_wrap import export_rfdetr_onnx

        onnx_path, imgsz = export_rfdetr_onnx(weights, out_dir / "model.onnx")
    log(f"exported {onnx_path}")
    calib = calibration_batches(calib_paths, imgsz, fmt, limit=export_cfg["calibration_limit"])

    artifacts = {"torch_fp32": weights, "onnx_fp32": onnx_path}
    artifacts["openvino_fp16"] = to_openvino(onnx_path, out_dir / "openvino", "fp16")
    artifacts["openvino_int8"] = to_openvino(onnx_path, out_dir / "openvino", "int8", calib, transformer=(family == "rfdetr"))
    try:
        artifacts["onnx_int8"] = onnx_int8(onnx_path, out_dir / "model.int8.onnx", calib, keep_head_fp=(family == "yolo"))
    except Exception as e:  # ORT static quantisation can fail on some transformer graphs; report, don't crash
        log(f"ONNX INT8 failed: {e}")
    try:
        artifacts["onnx_fp16_web"] = onnx_fp16(onnx_path, out_dir / "model.fp16.onnx")
    except Exception as e:
        log(f"ONNX FP16 failed: {e}")

    sample = read_rgb(test_paths[0])
    rows = {}
    base = None
    for name, path in artifacts.items():
        if name == "torch_fp32":
            det = load_detector(path, imgsz, fmt)
        elif str(path).endswith(".xml"):
            det = OpenVINODetector(path, imgsz, fmt)
        else:
            det = OnnxDetector(path, imgsz, fmt)
        preds = []
        for i in range(0, len(test_paths), 8):
            preds.extend(det.predict([read_rgb(p) for p in test_paths[i : i + 8]]))
        data = M.EvalData(test_data.labels, test_data.types, preds, test_data.gts)
        acc = M.compute(data)
        if base is None:
            base = acc
        lat = latency(det, sample, export_cfg["bench_warmup"], export_cfg["bench_runs"])
        rows[name] = {
            "path": str(path),
            "size_mb": round(size_mb(path), 2),
            **{k: round(v, 4) for k, v in acc.items()},
            **{f"d_{k}": round(acc[k] - base[k], 4) for k in acc},
            **{k: round(v, 2) for k, v in lat.items()},
        }
        log(f"{name}: {rows[name]}")
    gate = export_cfg["int8_max_ap_drop"]
    int8 = rows.get("openvino_int8")
    report = {
        "machine": machine(),
        "imgsz": imgsz,
        "format": fmt,
        "runtimes": rows,
        "release_gate": {
            "rule": f"OpenVINO INT8 image AP drop <= {gate}",
            "passed": bool(int8 is not None and -int8["d_image_ap"] <= gate),
        },
    }
    write_json(out_dir / "report.json", report)
    return report
