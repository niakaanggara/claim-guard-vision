"""Synthetic Indonesian workshop receipt generation (PDPA-compliant, no real PII)."""

from src.synthetic.gen_receipts import Receipt, generate_receipt, render_receipt_image

__all__ = ["Receipt", "generate_receipt", "render_receipt_image"]
