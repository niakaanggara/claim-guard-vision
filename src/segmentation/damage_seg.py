"""Damage instance-segmentation inference wrapper.

Supports two backends, chosen automatically unless ``backend`` is given:

* ``ultralytics`` — a YOLOv8-seg ``.pt`` model via the ``ultralytics``
  package (``YOLO(weights).predict(...)``).
* ``onnx`` — a YOLOv8-seg ``.onnx`` model via ONNX Runtime. The ONNX
  output is decoded generically: the first output is expected to be the
  YOLO detection tensor ``(1, 4 + nc + nm, N)`` and the second (when
  present) the prototype masks ``(1, nm, H, W)``.

If neither backend / weights file is available, :meth:`DamageSegmenter.predict`
raises a clear :class:`RuntimeError` naming what is missing — it never
silently returns fake detections. The dashboard catches this and falls
back to an ELA-only view.

Class names come from ``configs/model_config.yaml``; the defaults here
mirror the House-Tree-Person / damage vocabulary used in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

ImageLike = Union[str, Path, Image.Image, np.ndarray]

DEFAULT_CLASS_NAMES: dict[int, str] = {0: "house", 1: "tree", 2: "person"}


@dataclass
class SegmentationResult:
    """One image's instance-segmentation output."""

    masks: list[np.ndarray] = field(default_factory=list)  # binary (H, W) per instance
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)  # (height, width)
    backend: str = "unknown"

    def __len__(self) -> int:
        return len(self.masks)

    def area_ratios(self, reference_area: float) -> list[float]:
        """Return ``r_i`` for every mask (see :mod:`src.segmentation.geometry`)."""
        from src.segmentation.geometry import mask_area_ratio

        return [mask_area_ratio(m, reference_area) for m in self.masks]


class DamageSegmenter:
    """Thin wrapper over Ultralytics YOLOv8-seg or an ONNX segmentation model."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        onnx_path: Optional[Union[str, Path]] = None,
        backend: str = "auto",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[dict[int, str]] = None,
    ) -> None:
        self.weights = Path(weights) if weights else None
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.class_names = dict(class_names) if class_names else dict(DEFAULT_CLASS_NAMES)
        self._backend = self._resolve_backend(backend)
        self._model = None
        self._session = None
        self._load()

    # -- setup -------------------------------------------------------------
    def _resolve_backend(self, backend: str) -> str:
        backend = backend.lower()
        if backend == "auto":
            if self.weights is not None and self.weights.suffix == ".pt":
                return "ultralytics"
            if self.onnx_path is not None and self.onnx_path.suffix == ".onnx":
                return "onnx"
            if self.weights is not None and self.weights.suffix == ".onnx":
                self.onnx_path = self.weights
                return "onnx"
            return "none"
        if backend not in ("ultralytics", "onnx"):
            raise ValueError(f"backend must be auto | ultralytics | onnx, got {backend!r}")
        return backend

    def _load(self) -> None:
        if self._backend == "ultralytics":
            if self.weights is None or not self.weights.exists():
                raise FileNotFoundError(
                    f"YOLO weights not found at {self.weights}. "
                    "Download e.g. yolov8s-seg.pt or point `weights` at your model."
                )
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        elif self._backend == "onnx":
            path = self.onnx_path or self.weights
            if path is None or not path.exists():
                raise FileNotFoundError(f"ONNX model not found at {path}.")
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        # backend == "none": lazy error is raised in predict(), so importing
        # the module and constructing the object stay cheap in tests.

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_available(self) -> bool:
        return self._backend in ("ultralytics", "onnx")

    # -- inference ---------------------------------------------------------
    @staticmethod
    def _load_image(image: ImageLike) -> tuple[Image.Image, tuple[int, int]]:
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            pil = Image.open(image).convert("RGB")
        return pil, (pil.height, pil.width)

    def predict(self, image: ImageLike) -> SegmentationResult:
        """Run segmentation and return masks, boxes, scores, and classes."""
        pil, shape = self._load_image(image)
        if self._backend == "ultralytics":
            return self._predict_ultralytics(pil, shape)
        if self._backend == "onnx":
            return self._predict_onnx(pil, shape)
        raise RuntimeError(
            "No segmentation model is available: set `weights` to a YOLOv8-seg "
            ".pt file or `onnx_path` to a .onnx file (see configs/model_config.yaml)."
        )

    def _predict_ultralytics(self, pil: Image.Image, shape: tuple[int, int]) -> SegmentationResult:
        assert self._model is not None
        results = self._model.predict(
            pil, conf=self.confidence, iou=self.iou_threshold, verbose=False
        )
        if not results:
            return SegmentationResult(image_shape=shape, backend="ultralytics")
        r = results[0]
        masks, boxes, scores, class_ids, names = [], [], [], [], []
        if r.masks is not None:
            for mask_arr in r.masks.data.cpu().numpy():
                masks.append((mask_arr > 0.5).astype(np.uint8))
        if r.boxes is not None:
            for box, score, cls in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
            ):
                boxes.append(tuple(float(v) for v in box))
                scores.append(float(score))
                cid = int(cls)
                class_ids.append(cid)
                names.append(self.class_names.get(cid, str(cid)))
        # Masks and boxes are aligned by the Ultralytics API; pad defensively.
        n = max(len(masks), len(boxes))
        return SegmentationResult(
            masks=masks[:n], boxes=boxes[:n], scores=scores[:n],
            class_ids=class_ids[:n], class_names=names[:n],
            image_shape=shape, backend="ultralytics",
        )

    def _predict_onnx(self, pil: Image.Image, shape: tuple[int, int]) -> SegmentationResult:
        """Decode a YOLOv8-seg ONNX output (detections + prototype masks)."""
        assert self._session is not None
        import cv2

        input_name = self._session.get_inputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        target_h, target_w = int(input_shape[2]), int(input_shape[3])
        resized = pil.resize((target_w, target_h), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))[None, :, :, :].astype(np.float32)
        outputs = self._session.run(None, {input_name: tensor})
        det = np.asarray(outputs[0])  # (1, 4+nc+nm, N) or (1, N, 4+nc+nm)
        if det.ndim == 3 and det.shape[1] > det.shape[2]:
            det = np.transpose(det, (0, 2, 1))
        det = det[0]  # (N, 4+nc+nm)
        if det.shape[0] == 0:
            return SegmentationResult(image_shape=shape, backend="onnx")

        boxes_xywh = det[:, :4]
        scores_all = det[:, 4:]
        # YOLOv8-seg ONNX: columns 4..4+nc-1 are class scores, rest are mask coeffs.
        nc = max(1, scores_all.shape[1] - 32)
        class_scores = scores_all[:, :nc]
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)
        keep = scores >= self.confidence
        if not np.any(keep):
            return SegmentationResult(image_shape=shape, backend="onnx")

        h, w = shape
        sx, sy = w / target_w, h / target_h
        masks, boxes, out_scores, out_ids, out_names = [], [], [], [], []
        for idx in np.where(keep)[0]:
            cx, cy, bw, bh = boxes_xywh[idx]
            x1, y1 = (cx - bw / 2) * sx, (cy - bh / 2) * sy
            x2, y2 = (cx + bw / 2) * sx, (cy + bh / 2) * sy
            boxes.append((float(x1), float(y1), float(x2), float(y2)))
            out_scores.append(float(scores[idx]))
            cid = int(class_ids[idx])
            out_ids.append(cid)
            out_names.append(self.class_names.get(cid, str(cid)))
            # Mask: sigmoid(coeffs @ prototypes), cropped to the box.
            if len(outputs) > 1:
                protos = np.asarray(outputs[1])[0]  # (nm, ph, pw)
                coeffs = scores_all[idx, nc:]
                mask_small = 1.0 / (1.0 + np.exp(-(coeffs @ protos.reshape(protos.shape[0], -1)).reshape(protos.shape[1:])))
                mask_full = cv2.resize(
                    (mask_small > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
                masks.append(mask_full)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))] = 1
                masks.append(mask)
        return SegmentationResult(
            masks=masks, boxes=boxes, scores=out_scores,
            class_ids=out_ids, class_names=out_names,
            image_shape=shape, backend="onnx",
        )
