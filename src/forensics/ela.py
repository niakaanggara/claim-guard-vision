"""Error Level Analysis (ELA) for forensic image tampering detection.

Idea
----
A JPEG image is a lossy compression of the original pixels. If an image is
re-saved at a known quality factor, regions that were already compressed at
that quality change very little, while regions that were edited, pasted in,
or re-compressed at a different quality show a much larger error.

ELA therefore:
1. loads the image,
2. re-saves it as JPEG at ``quality``,
3. takes the per-pixel absolute difference between the original and the
   re-compressed copy,
4. amplifies that difference into a heatmap, and
5. summarises it as a tampering probability ``P_tamper in [0, 1]``.

The score is a *heuristic*, not a proof of tampering: high-frequency texture,
noise, and legitimate re-encoding also raise it. Always combine it with the
segmentation / actuarial stages and, where available, the TamperNet model
(see :mod:`src.forensics.tamper_net`).

References: Krawetz, "A Picture's Worth" (FotoForensics ELA).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

try:  # OpenCV is optional for the heatmap colourisation only.
    import cv2
except ImportError:  # pragma: no cover - exercised when cv2 is absent
    cv2 = None  # type: ignore[assignment]

ImageLike = Union[str, Path, Image.Image, np.ndarray]


@dataclass(frozen=True)
class ELAResult:
    """Result of one ELA computation."""

    diff: np.ndarray  # float32 difference map, shape (H, W), values in [0, 255]
    heatmap: np.ndarray  # uint8 RGB heatmap, shape (H, W, 3)
    mean_error: float
    std_error: float
    p95_error: float
    p99_error: float
    fraction_above_threshold: float
    p_tamper: float  # in [0, 1]

    def to_dict(self) -> dict:
        """Return the scalar statistics (not the arrays) as a plain dict."""
        return {
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "p95_error": self.p95_error,
            "p99_error": self.p99_error,
            "fraction_above_threshold": self.fraction_above_threshold,
            "p_tamper": self.p_tamper,
        }


def _load_rgb(image: ImageLike) -> Image.Image:
    """Load any supported input as an RGB Pillow image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return Image.open(image).convert("RGB")


def _recompress(image: Image.Image, quality: int) -> Image.Image:
    """Re-save ``image`` as JPEG at ``quality`` and reload it."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def ela_anomaly_score(
    diff: np.ndarray,
    *,
    threshold: float = 30.0,
    scale: float = 10.0,
) -> float:
    """Map an ELA difference map to ``P_tamper in [0, 1]``.

    The score blends two signals so that neither a few hot pixels nor a
    uniformly noisy image dominates on its own:

    * the *tail* of the error distribution (p95), which catches local edits;
    * the *fraction* of pixels above ``threshold``, which catches broad
      re-compression.

    Parameters
    ----------
    diff:
        Float difference map with values in [0, 255].
    threshold:
        Pixel error above which a pixel counts as anomalous.
    scale:
        Amplification used when the heatmap was built; kept for API
        symmetry and future calibration (currently unused in the score).

    Returns
    -------
    float
        Tampering probability in [0, 1].
    """
    if diff.size == 0:
        return 0.0
    flat = np.asarray(diff, dtype=np.float32).ravel()
    p95 = float(np.percentile(flat, 95))
    frac = float(np.mean(flat > threshold))
    # p95 / 255 in [0, 1]; frac in [0, 1]. Weight the tail more heavily.
    tail_component = 1.0 - math.exp(-p95 / 45.0)
    score = 0.65 * tail_component + 0.35 * frac
    return float(min(1.0, max(0.0, score)))


def _make_heatmap(diff: np.ndarray, scale: float) -> np.ndarray:
    """Amplify ``diff`` and colourise it as an RGB heatmap."""
    amplified = np.clip(diff * scale, 0, 255).astype(np.uint8)
    if cv2 is not None:
        coloured = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)
        return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    # Fallback without OpenCV: grayscale heatmap repeated over 3 channels.
    return np.stack([amplified] * 3, axis=-1)


def compute_ela(
    image: ImageLike,
    *,
    quality: int = 90,
    scale: float = 10.0,
    threshold: float = 30.0,
    max_dimension: int = 1024,
) -> ELAResult:
    """Compute the ELA heatmap and anomaly statistics for one image.

    Parameters
    ----------
    image:
        Path, Pillow image, or NumPy array.
    quality:
        JPEG quality (1-100) for the recompression step. 90 is the usual
        default; lower values make the analysis more sensitive.
    scale:
        Brightness amplification for the heatmap.
    threshold:
        Pixel error above which a pixel counts as anomalous.
    max_dimension:
        Images larger than this (on their longest side) are downscaled
        first, to keep the analysis fast and memory-bounded.

    Returns
    -------
    ELAResult
        Difference map, RGB heatmap, summary statistics, and ``p_tamper``.
    """
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    original = _load_rgb(image)
    if max(original.size) > max_dimension:
        original.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    recompressed = _recompress(original, quality=int(quality))
    orig_arr = np.asarray(original, dtype=np.float32)
    recomp_arr = np.asarray(recompressed, dtype=np.float32)

    # Mean absolute error across channels -> single-channel difference map.
    diff = np.mean(np.abs(orig_arr - recomp_arr), axis=2).astype(np.float32)

    if cv2 is not None and 3 > 0:
        diff = cv2.medianBlur(diff.astype(np.uint8), 3).astype(np.float32) if diff.shape[0] >= 3 and diff.shape[1] >= 3 else diff

    heatmap = _make_heatmap(diff, scale=float(scale))
    flat = diff.ravel()
    p_tamper = ela_anomaly_score(diff, threshold=threshold, scale=scale)

    return ELAResult(
        diff=diff,
        heatmap=heatmap,
        mean_error=float(np.mean(flat)),
        std_error=float(np.std(flat)),
        p95_error=float(np.percentile(flat, 95)),
        p99_error=float(np.percentile(flat, 99)),
        fraction_above_threshold=float(np.mean(flat > threshold)),
        p_tamper=float(p_tamper),
    )


def save_heatmap(result: ELAResult, path: Union[str, Path]) -> Path:
    """Save an :class:`ELAResult` heatmap to ``path`` and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.heatmap).save(out)
    return out
