"""Ship small: ONNX export, OpenVINO FP32/FP16/INT8, and ONNX Runtime INT8 for the browser.

INT8 is post-training quantisation calibrated on TRAINING images only (good
parts, the few real defects and, if given, synthetic defects). Following
Ultralytics' own OpenVINO exporter, the detection head is left in floating
point: quantising the box decoder and score sigmoid costs accuracy and buys
little speed, because the backbone and neck hold almost all the compute.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from ..detect.runtime import preprocess
from ..io import read_rgb


def export_yolo_onnx(weights: str | Path, out_path: str | Path, imgsz: int = 640, opset: int | None = None) -> Path:
    from ultralytics import YOLO

    from ..detect.yolo import _quiet_ultralytics

    _quiet_ultralytics()
    # nms=False selects YOLO26's NMS-free one-to-one head: output (1, 300, 6), no NMS needed anywhere,
    # which is what lets the browser demo decode results in a few lines of JavaScript.
    kwargs = dict(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, half=False, nms=False)
    if opset:
        kwargs["opset"] = opset
    produced = Path(YOLO(str(weights)).export(**kwargs))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), out_path)
    return out_path


def calibration_batches(image_paths: list[str], imgsz: int, fmt: str, limit: int = 300, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    paths = list(image_paths)
    if len(paths) > limit:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), limit, replace=False))]
    return [preprocess([read_rgb(p)], imgsz, fmt)[0] for p in paths]


def _head_scope_names(model) -> list[str]:
    """Names of the detection-head ops to keep in floating point (Ultralytics' rule, for ONNX-style names)."""
    ops = model.get_ordered_ops()
    sigmoids = [op.get_friendly_name() for op in ops if op.get_type_name() == "Sigmoid"]
    if not sigmoids:
        return []
    last = sigmoids[-1]
    parts = last.strip("/").split("/")
    prefix = ("/" + parts[0] + "/") if last.startswith("/") else (parts[0] + "/")
    return [op.get_friendly_name() for op in ops if op.get_type_name() == "Sigmoid" or op.get_friendly_name().startswith(prefix)]


def to_openvino(
    onnx_path: str | Path,
    out_dir: str | Path,
    precision: str,
    calib: list[np.ndarray] | None = None,
    transformer: bool = False,
    keep_head_fp: bool = True,
) -> Path:
    import openvino as ov

    if precision not in ("fp32", "fp16", "int8"):
        raise ValueError(precision)
    model = ov.convert_model(str(onnx_path))
    if precision == "int8":
        import nncf

        if not calib:
            raise ValueError("INT8 needs calibration images")
        ignored = None
        names = _head_scope_names(model) if keep_head_fp and not transformer else []
        if names:
            ignored = nncf.IgnoredScope(names=names, validate=False)
        model = nncf.quantize(
            model,
            nncf.Dataset(calib),
            preset=nncf.QuantizationPreset.MIXED,
            model_type=nncf.ModelType.TRANSFORMER if transformer else None,
            subset_size=len(calib),
            ignored_scope=ignored,
        )
    out = Path(out_dir) / precision / "model.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(out), compress_to_fp16=(precision == "fp16"))
    return out


def onnx_int8(onnx_path: str | Path, out_path: str | Path, calib: list[np.ndarray], keep_head_fp: bool = True) -> Path:
    """Static QDQ INT8 for ONNX Runtime (CPU and the browser's WASM backend)."""
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, name: str):
            self.it = iter([{name: x} for x in calib])

        def get_next(self):
            return next(self.it, None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pre = out_path.with_suffix(".pre.onnx")
    quant_pre_process(str(onnx_path), str(pre), skip_symbolic_shape=True)
    m = onnx.load(str(pre))
    name = m.graph.input[0].name
    exclude = []
    if keep_head_fp:
        sig = [n.name for n in m.graph.node if n.op_type == "Sigmoid"]
        if sig:
            prefix = "/" + sig[-1].strip("/").split("/")[0] + "/"
            exclude = [n.name for n in m.graph.node if n.name.startswith(prefix) or n.op_type == "Sigmoid"]
    quantize_static(
        str(pre),
        str(out_path),
        Reader(name),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=exclude,
    )
    pre.unlink(missing_ok=True)
    return out_path


def _head_nodes(m) -> list[str]:
    sig = [n.name for n in m.graph.node if n.op_type == "Sigmoid"]
    if not sig:
        return []
    prefix = "/" + sig[-1].strip("/").split("/")[0] + "/"
    return [n.name for n in m.graph.node if n.name.startswith(prefix)]


def onnx_fp16(onnx_path: str | Path, out_path: str | Path) -> Path:
    """FP16 weights with FP32 inputs/outputs (half the download for the browser demo).

    The detection head (box decoding, top-k, casts) stays FP32: converting it
    produces a graph ONNX Runtime refuses to load (a Cast type mismatch, caught
    by the pipeline test). The result is loaded once here so a bad file never ships.
    """
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    src = onnx.load(str(onnx_path))
    m = float16.convert_float_to_float16(src, keep_io_types=True, node_block_list=_head_nodes(src))
    out_path = Path(out_path)
    onnx.save(m, str(out_path))
    ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    return out_path
