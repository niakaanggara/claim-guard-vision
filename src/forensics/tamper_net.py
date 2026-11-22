"""Lightweight CNN / ONNX wrapper for tampering-probability inference.

``TamperNet`` is the learned counterpart to ELA (see
:mod:`src.forensics.ela`): where ELA is a hand-crafted compression
heuristic, this module runs a small classifier — exported to ONNX — that
outputs ``P_tamper in [0, 1]`` directly from the pixels.

Two modes are supported:

* **ONNX mode (recommended for inference):** pass ``onnx_path`` to a
  ``.onnx`` file and the model is run with ONNX Runtime (CPU-friendly,
  no PyTorch needed at inference time).
* **ELA fallback:** if no ONNX file is available, :meth:`TamperNet.predict`
  falls back to the ELA anomaly score so the pipeline still produces a
  calibrated ``[0, 1]`` probability instead of crashing. The fallback is
  clearly flagged on the returned prediction.

The PyTorch model definition (:class:`TamperCNN`) is included so the same
architecture can be trained and exported with ``src/export_onnx.py``.
Importing this module does *not* require PyTorch — it is imported lazily
only when the training/export path is used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

from src.forensics.ela import compute_ela

ImageLike = Union[str, Path, Image.Image, np.ndarray]


@dataclass(frozen=True)
class TamperPrediction:
    """One tampering-probability prediction."""

    p_tamper: float  # in [0, 1]
    source: str  # "onnx" or "ela_fallback"
    model_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_tamper <= 1.0:
            raise ValueError(f"p_tamper must be in [0, 1], got {self.p_tamper}")


class TamperCNN:  # pragma: no cover - requires torch; exercised via export_onnx.py
    """Small CNN for tampering classification (training / export only).

    Architecture: 3 convolutional blocks (conv -> BN -> ReLU -> max-pool)
    followed by global average pooling and a single-logit head. Input is a
    224x224 RGB image; output is a single logit (sigmoid -> P_tamper).
    """

    def __init__(self, num_classes: int = 1) -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def __call__(self, x):  # pragma: no cover
        return self.model(x)


def _preprocess(image: ImageLike, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize, normalise (ImageNet), and batch an image for ONNX inference."""
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.open(image).convert("RGB")
    pil = pil.resize(size, Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)  # (1, 3, H, W)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class TamperNet:
    """ONNX inference wrapper with an ELA fallback.

    Parameters
    ----------
    onnx_path:
        Path to an exported ``.onnx`` TamperNet model. If ``None`` or the
        file does not exist, every prediction uses the ELA fallback.
    input_size:
        Model input size as ``(height, width)``.
    """

    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        input_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.input_size = input_size
        self._session = None
        self._input_name: Optional[str] = None

        if self.onnx_path is not None and self.onnx_path.exists():
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.onnx_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        elif self.onnx_path is not None:
            warnings.warn(
                f"TamperNet ONNX model not found at {self.onnx_path}; "
                "predictions will use the ELA fallback.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def backend(self) -> str:
        """Return ``"onnx"`` when a model is loaded, else ``"ela_fallback"``."""
        return "onnx" if self._session is not None else "ela_fallback"

    def predict(self, image: ImageLike) -> TamperPrediction:
        """Return ``P_tamper`` for a single image."""
        if self._session is not None:
            assert self._input_name is not None
            tensor = _preprocess(image, size=self.input_size)
            outputs = self._session.run(None, {self._input_name: tensor})
            raw = np.asarray(outputs[0]).ravel()
            # TamperCNN exports a single logit; sigmoid maps it to [0, 1].
            # If a model already exports a probability, sigmoid is still
            # monotonic, so ranking is preserved; export a logit head
            # (see export_onnx.py) for calibrated probabilities.
            p = float(min(1.0, max(0.0, _sigmoid(float(raw[0])))))
            return TamperPrediction(p_tamper=p, source="onnx", model_path=str(self.onnx_path))
        result = compute_ela(image)
        return TamperPrediction(p_tamper=result.p_tamper, source="ela_fallback")

    def predict_batch(self, images: Sequence[ImageLike]) -> list[TamperPrediction]:
        """Return a prediction for each image in ``images``."""
        return [self.predict(img) for img in images]
