"""Mask / polygon geometry: areas, area ratios ``r_i``, and IoU metrics.

The actuarial stage needs one number per damaged component:

    r_i = A_mask,i / A_reference

where ``A_mask,i`` is the pixel area of the i-th damage mask and
``A_reference`` is the area of a reference object in the same photo
(for example the whole vehicle, house facade, or image area). ``r_i`` is
dimensionless and in ``[0, 1]`` when the mask lies inside the reference.

All functions here are pure NumPy (OpenCV is used only for contour
extraction, with a NumPy fallback) so they are easy to unit-test.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def mask_area(mask: np.ndarray) -> float:
    """Return the pixel area of a binary mask (non-zero pixels)."""
    arr = np.asarray(mask)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr))


def mask_area_ratio(mask: np.ndarray, reference_area: float) -> float:
    """Return ``r = A_mask / A_reference``, clamped to ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``reference_area`` is not positive.
    """
    if reference_area <= 0:
        raise ValueError(f"reference_area must be positive, got {reference_area}")
    return float(min(1.0, max(0.0, mask_area(mask) / float(reference_area))))


def polygon_area(points: np.ndarray) -> float:
    """Return the area of a polygon via the shoelace formula.

    Parameters
    ----------
    points:
        Array of shape ``(N, 2)`` with ``(x, y)`` vertices in order.
        Returns 0.0 for fewer than 3 points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Extract external contours of a binary mask as ``(N, 2)`` polygons.

    Uses OpenCV when available; otherwise returns an empty list (the
    mask itself is still usable via :func:`mask_area`).
    """
    if cv2 is None:
        return []
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(x_min, y_min, x_max, y_max)`` of the non-zero mask region."""
    arr = np.asarray(mask)
    ys, xs = np.nonzero(arr)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def iou_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return the Intersection-over-Union of two binary masks in ``[0, 1]``.

    Returns 0.0 when both masks are empty, and 1.0 when both are empty
    is *not* used — empty/empty is defined as 0.0 here so that a missing
    prediction never scores perfectly. Callers that need the other
    convention should handle the empty case explicitly.
    """
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    if a.shape != b.shape:
        raise ValueError(f"mask shapes must match, got {a.shape} vs {b.shape}")
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    return inter / union if union > 0 else 0.0


def dice_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return the Dice coefficient of two binary masks in ``[0, 1]``."""
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    if a.shape != b.shape:
        raise ValueError(f"mask shapes must match, got {a.shape} vs {b.shape}")
    inter = float(np.count_nonzero(a & b))
    denom = float(np.count_nonzero(a) + np.count_nonzero(b))
    return (2.0 * inter / denom) if denom > 0 else 0.0


def iou_boxes(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Return the IoU of two ``(x_min, y_min, x_max, y_max)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
