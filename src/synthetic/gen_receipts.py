"""Synthetic Indonesian workshop repair receipt generator.

Generates **fully synthetic** receipts for document-verification testing:
no real customer, workshop, plate, or invoice data is used. Names,
addresses, and phone numbers come from ``Faker('id_ID')`` with a fixed
seed, so a dataset is reproducible. This keeps generated data
PDPA-compliant by construction — nothing in it identifies a real person.

Each receipt carries:

* a workshop header (name, Indonesian address, phone),
* an invoice number, date, and Indonesian vehicle plate (e.g. ``B 1234 ABC``),
* line items (spare parts + labour) with IDR prices,
* subtotal, PPN 11%, and grand total,
* a rendered PNG image (Pillow) and a JSON annotation file with a
  bounding box per text line, for OCR / document-verification tests.

Example
-------
>>> receipt = generate_receipt(seed=42)
>>> receipt.total_idr > 0
True
>>> path, ann = render_receipt_image(receipt, "out/receipt_0001.png")
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# Typical Indonesian workshop line items: (description, price range in IDR).
PARTS: list[tuple[str, tuple[int, int]]] = [
    ("Oli Mesin 10W-40 (4L)", (350_000, 650_000)),
    ("Filter Oli", (45_000, 95_000)),
    ("Kampas Rem Depan", (250_000, 550_000)),
    ("Kampas Rem Belakang", (200_000, 450_000)),
    ("Busi Iridium (4 pcs)", (180_000, 360_000)),
    ("Aki 12V 45Ah", (850_000, 1_450_000)),
    ("Ban 205/55 R16", (750_000, 1_250_000)),
    ("Lampu Depan LED", (300_000, 800_000)),
    ("Kaca Spion Kanan", (275_000, 650_000)),
    ("Bumper Depan", (900_000, 2_200_000)),
    ("Pintu Kanan (cat ulang)", (1_200_000, 2_800_000)),
    ("Radiator Coolant (4L)", (120_000, 220_000)),
]

LABOUR: list[tuple[str, tuple[int, int]]] = [
    ("Jasa Servis Berkala", (150_000, 300_000)),
    ("Jasa Ganti Oli", (75_000, 150_000)),
    ("Jasa Perbaikan Rem", (150_000, 350_000)),
    ("Jasa Cat & Body Repair", (500_000, 1_500_000)),
    ("Jasa Spooring & Balancing", (200_000, 400_000)),
    ("Jasa Tune Up Mesin", (250_000, 500_000)),
]

PPN_RATE = 0.11  # Indonesian VAT (PPN) rate.
VEHICLE_BRANDS = ["Toyota Avanza", "Honda Brio", "Daihatsu Xenia", "Suzuki Ertiga", "Mitsubishi Xpander"]
PLATE_PREFIXES = ["B", "D", "L", "AB", "AA", "H", "F"]


def _idr(amount: int) -> str:
    """Format an integer as Indonesian rupiah, e.g. ``Rp1.250.000``."""
    return f"Rp{amount:,}".replace(",", ".")


@dataclass
class ReceiptItem:
    description: str
    qty: int
    unit_price_idr: int

    @property
    def line_total_idr(self) -> int:
        return self.qty * self.unit_price_idr


@dataclass
class Receipt:
    invoice_no: str
    date: str  # ISO date, e.g. "2026-11-22"
    workshop_name: str
    workshop_address: str
    workshop_phone: str
    customer_name: str
    vehicle: str
    plate_no: str
    items: list[ReceiptItem] = field(default_factory=list)

    @property
    def subtotal_idr(self) -> int:
        return sum(item.line_total_idr for item in self.items)

    @property
    def ppn_idr(self) -> int:
        return int(round(self.subtotal_idr * PPN_RATE))

    @property
    def total_idr(self) -> int:
        return self.subtotal_idr + self.ppn_idr

    def to_dict(self) -> dict:
        data = asdict(self)
        data["subtotal_idr"] = self.subtotal_idr
        data["ppn_idr"] = self.ppn_idr
        data["total_idr"] = self.total_idr
        return data


def _plate(rng: random.Random) -> str:
    return f"{rng.choice(PLATE_PREFIXES)} {rng.randint(1000, 9999)} {rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}"


def generate_receipt(seed: Optional[int] = None) -> Receipt:
    """Generate one synthetic receipt. ``seed`` makes the output reproducible."""
    rng = random.Random(seed)
    fake = Faker("id_ID")
    if seed is not None:
        Faker.seed(seed)

    n_parts = rng.randint(2, 4)
    n_labour = rng.randint(1, 2)
    items: list[ReceiptItem] = []
    for desc, (low, high) in rng.sample(PARTS, k=n_parts):
        items.append(ReceiptItem(desc, rng.randint(1, 2), rng.randint(low, high) // 1000 * 1000))
    for desc, (low, high) in rng.sample(LABOUR, k=n_labour):
        items.append(ReceiptItem(desc, 1, rng.randint(low, high) // 1000 * 1000))

    receipt_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 320))
    return Receipt(
        invoice_no=f"INV/{receipt_date.year}/{receipt_date.month:02d}/{rng.randint(1000, 9999)}",
        date=receipt_date.isoformat(),
        workshop_name=f"Bengkel {fake.last_name()} Motor",
        workshop_address=fake.address().replace("\n", ", "),
        workshop_phone=fake.phone_number(),
        customer_name=fake.name(),
        vehicle=rng.choice(VEHICLE_BRANDS),
        plate_no=_plate(rng),
        items=items,
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_receipt_image(
    receipt: Receipt,
    output_path: str | Path,
    width: int = 640,
) -> tuple[Path, dict]:
    """Render ``receipt`` as a PNG and return ``(image_path, annotations)``.

    The annotation dict follows a simple COCO-style line format::

        {"image": "receipt_0001.png", "lines": [{"text": ..., "bbox": [x, y, w, h]}, ...]}

    where ``bbox`` is in pixels, ``[x, y, width, height]``.
    A ``.json`` file with the same stem is written next to the image.
    """
    font_title = _load_font(22)
    font_body = _load_font(16)
    font_small = _load_font(14)

    lines: list[tuple[str, object]] = [
        (receipt.workshop_name.upper(), font_title),
        (receipt.workshop_address, font_small),
        (f"Telp: {receipt.workshop_phone}", font_small),
        ("-" * 48, font_body),
        (f"No. Invoice : {receipt.invoice_no}", font_body),
        (f"Tanggal     : {receipt.date}", font_body),
        (f"Pelanggan   : {receipt.customer_name}", font_body),
        (f"Kendaraan   : {receipt.vehicle}  ({receipt.plate_no})", font_body),
        ("-" * 48, font_body),
    ]
    for item in receipt.items:
        lines.append((f"{item.description}", font_body))
        lines.append((f"  {item.qty} x {_idr(item.unit_price_idr)} = {_idr(item.line_total_idr)}", font_body))
    lines += [
        ("-" * 48, font_body),
        (f"Subtotal : {_idr(receipt.subtotal_idr)}", font_body),
        (f"PPN 11%  : {_idr(receipt.ppn_idr)}", font_body),
        (f"TOTAL    : {_idr(receipt.total_idr)}", font_title),
        ("Terima kasih atas kunjungan Anda.", font_small),
    ]

    line_height = 28
    height = 40 + line_height * len(lines) + 30
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    annotations: list[dict] = []
    y = 20
    for text, font in lines:
        draw.text((30, y), text, fill="black", font=font)  # type: ignore[arg-type]
        bbox = draw.textbbox((30, y), text, font=font)  # type: ignore[arg-type]
        annotations.append({
            "text": text,
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])],
        })
        y += line_height

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, format="PNG")

    annotation = {
        "image": out.name,
        "width": width,
        "height": height,
        "receipt": receipt.to_dict(),
        "lines": annotations,
    }
    ann_path = out.with_suffix(".json")
    ann_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, annotation


def generate_dataset(n: int, output_dir: str | Path, seed: int = 0) -> list[Path]:
    """Generate ``n`` synthetic receipts (PNG + JSON) into ``output_dir``."""
    out_dir = Path(output_dir)
    paths: list[Path] = []
    for i in range(n):
        receipt = generate_receipt(seed=seed + i)
        path, _ = render_receipt_image(receipt, out_dir / f"receipt_{i+1:04d}.png")
        paths.append(path)
    return paths


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Indonesian workshop receipts.")
    parser.add_argument("-n", "--count", type=int, default=5)
    parser.add_argument("-o", "--output-dir", default="data/samples/receipts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    made = generate_dataset(args.count, args.output_dir, seed=args.seed)
    print(f"Generated {len(made)} receipts in {args.output_dir}")
