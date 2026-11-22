"""Damage instance segmentation and mask geometry."""

from src.segmentation.damage_seg import DamageSegmenter, SegmentationResult
from src.segmentation.geometry import (
    iou_boxes,
    iou_masks,
    mask_area,
    mask_area_ratio,
    polygon_area,
)

__all__ = [
    "DamageSegmenter",
    "SegmentationResult",
    "iou_boxes",
    "iou_masks",
    "mask_area",
    "mask_area_ratio",
    "polygon_area",
]
