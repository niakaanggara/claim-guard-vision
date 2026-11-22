"""Export trained PyTorch / YOLO weights to ONNX.

Usage
-----
Export a YOLOv8-seg model::

    python -m src.export_onnx --model-type yolo \\
        --weights models/yolov8s-seg.pt --output models/yolov8s-seg.onnx

Export the TamperNet classifier (``src/forensics/tamper_net.py``)::

    python -m src.export_onnx --model-type tampernet \\
        --weights models/tamper_net.pt --output models/tamper_net.onnx

The exported file is validated by loading it with ONNX Runtime and
running one dummy forward pass, so a broken export fails loudly here
instead of at inference time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def export_yolo(weights: Path, output: Path, imgsz: int, opset: int, dynamic: bool) -> Path:
    """Export a YOLO model with the Ultralytics exporter."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, dynamic=dynamic)
    exported_path = Path(exported)
    if exported_path.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(output)
    return output


def export_tampernet(weights: Path, output: Path, opset: int, dynamic: bool) -> Path:
    """Export a TamperNet checkpoint with ``torch.onnx.export``."""
    import torch

    from src.forensics.tamper_net import TamperCNN

    wrapper = TamperCNN()
    state = torch.load(weights, map_location="cpu")
    # Accept both a raw state_dict and a {"state_dict": ...} checkpoint.
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    wrapper.model.load_state_dict(state_dict)  # type: ignore[attr-defined]
    wrapper.model.eval()  # type: ignore[attr-defined]

    dummy = torch.randn(1, 3, 224, 224)
    output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        wrapper.model,  # type: ignore[arg-type]
        dummy,
        str(output),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
    )
    return output


def validate_onnx(output: Path, input_shape: tuple[int, int, int, int] = (1, 3, 224, 224)) -> None:
    """Load ``output`` with ONNX Runtime and run one dummy forward pass."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    dummy = np.random.rand(*input_shape).astype(np.float32)
    result = session.run(None, {name: dummy})
    print(f"ONNX validation OK: {output} -> output shape {np.asarray(result[0]).shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch/YOLO weights to ONNX.")
    parser.add_argument("--model-type", choices=["yolo", "tampernet"], required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Input .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch axis")
    parser.add_argument("--no-validate", action="store_true", help="Skip ONNX Runtime validation")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.output.suffix != ".onnx":
        raise ValueError(f"Output must end with .onnx, got {args.output}")

    if args.model_type == "yolo":
        exported = export_yolo(args.weights, args.output, args.imgsz, args.opset, args.dynamic)
        shape = (1, 3, args.imgsz, args.imgsz)
    else:
        exported = export_tampernet(args.weights, args.output, args.opset, args.dynamic)
        shape = (1, 3, 224, 224)

    print(f"Exported {args.weights} -> {exported}")
    if not args.no_validate:
        validate_onnx(exported, input_shape=shape)


if __name__ == "__main__":
    main()
