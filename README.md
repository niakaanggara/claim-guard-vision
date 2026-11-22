# Claim Guard Vision

**Forensic Image Tampering Detection & Actuarial Claim Severity Estimation** — an end-to-end Computer Vision pipeline for Insurance & Financial Audit automation.

Upload a claim photo, and Claim Guard Vision answers three questions in one pass:

1. **Was this photo manipulated?** — Error Level Analysis (ELA) + a learned tamper classifier produce `P_tamper ∈ [0, 1]`.
2. **How much damage is visible?** — Instance segmentation measures each damaged component's area ratio `r_i = A_mask,i / A_reference`.
3. **What should happen to this claim?** — An actuarial engine turns those ratios into a severity index `S_claim` and triages the claim into **Reject / Forensic Audit**, **Manual Adjuster Review**, or **Auto-Approve (Fast Track)**.

> ⚠️ This is a portfolio / research scaffold. The forensics score is a heuristic signal, not legal proof of fraud, and the actuarial weights are illustrative — calibrate `configs/actuarial_weights.yaml` on historical claims before any production use.

---

## Architecture

```text
claim photo
    │
    ├─► src/forensics/ela.py          ELA heatmap + P_tamper (heuristic)
    │   └─► src/forensics/tamper_net.py   TamperNet (ONNX) — learned P_tamper
    │
    ├─► src/segmentation/damage_seg.py    YOLOv8-seg / ONNX inference
    │   └─► src/segmentation/geometry.py  mask area, r_i, IoU / Dice
    │
    ├─► src/actuarial/risk_engine.py      S_claim = Σ w_i · r_i^γ  + triage
    │
    └─► src/synthetic/gen_receipts.py     synthetic id_ID workshop receipts
                                          (Faker + Pillow, no real PII)

app/dashboard.py  —  Streamlit demo: Original | ELA Heatmap | Segmentation + Decision
```

### 1. Pixel-Level Forensics — `src/forensics/`

**Error Level Analysis** (`ela.py`): the image is re-saved as JPEG at a known quality factor; regions edited or re-compressed at a different quality show a much larger error than untouched regions. `compute_ela()` returns the difference map, a colourised heatmap, summary statistics (mean / std / p95 / p99 error, fraction of anomalous pixels), and `P_tamper`.

**TamperNet** (`tamper_net.py`): a lightweight CNN (3 conv blocks + global average pooling + single-logit head) exported to ONNX and run with ONNX Runtime on CPU. If no ONNX file is present, predictions fall back to the ELA score and say so explicitly (`source="ela_fallback"`).

### 2. Damage Segmentation & Geometry — `src/segmentation/`

`damage_seg.py` wraps **Ultralytics YOLOv8-seg** (`.pt`) and **ONNX Runtime** (`.onnx`) behind one `DamageSegmenter.predict()` API, returning masks, boxes, scores, and class names.

`geometry.py` provides the pure-NumPy math the actuarial stage depends on:

- `mask_area`, `polygon_area` (shoelace formula), `mask_to_polygons`
- `mask_area_ratio(mask, reference_area)` → `r_i ∈ [0, 1]`
- `iou_masks`, `dice_masks`, `iou_boxes`, `bounding_box`

### 3. Actuarial Risk & Triage — `src/actuarial/`

`risk_engine.py` implements:

```text
S_claim = Σᵢ wᵢ · rᵢ^γ          γ = 1.5 (default), so large concentrated
                                 damage counts more than the same area
                                 spread across many small components
```

Triage rules, evaluated in order (fraud always wins):

| Condition | Decision |
|---|---|
| `P_tamper ≥ τ_fraud` (default 0.65) | **REJECT / FORENSIC AUDIT** |
| `S_claim ≥ τ_cost` (default 0.45) | **MANUAL ADJUSTER REVIEW** |
| otherwise | **AUTO-APPROVE (FAST TRACK)** |

Weights, `γ`, and both thresholds live in `configs/actuarial_weights.yaml`; image/model settings live in `configs/model_config.yaml`. See `tests/test_risk_engine.py` for the full specification, including boundary cases (thresholds are inclusive).

### 4. Synthetic Indonesian Receipts — `src/synthetic/`

`gen_receipts.py` generates **fully synthetic** Indonesian workshop repair receipts with `Faker('id_ID')` — workshop name/address/phone, invoice number, date, customer, vehicle and plate (`B 1234 ABC`), spare-part + labour line items in IDR, PPN 11%, and totals — rendered as PNGs with Pillow plus a JSON annotation file containing a bounding box per text line, for OCR / document-verification testing. Nothing in the output identifies a real person, so the data is PDPA-compliant by construction.

```bash
python -m src.synthetic.gen_receipts -n 10 -o data/samples/receipts --seed 0
```

---

## Quickstart

```bash
git clone https://github.com/niakaanggara/claim-guard-vision.git
cd claim-guard-vision
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the tests
pytest -q

# Try the forensics + actuarial stages on a photo (no model weights needed)
python -c "
from src.forensics.ela import compute_ela
from src.actuarial.risk_engine import RiskEngine, DamageComponent, load_actuarial_config
ela = compute_ela('data/samples/your-photo.jpg')
engine = RiskEngine(load_actuarial_config())
print(engine.assess([DamageComponent('vehicle_body', 0.12)], p_tamper=ela.p_tamper))
"

# Launch the interactive demo
streamlit run app/dashboard.py
```

The demo works without model weights: ELA and the actuarial engine run on CPU, and segmentation degrades gracefully to an ELA-only view when no YOLO/ONNX weights are configured.

### Adding segmentation weights (optional)

Place weights under `models/` (git-ignored) and point `configs/model_config.yaml` at them:

```yaml
segmentation:
  weights: "models/yolov8s-seg.pt"      # Ultralytics
  onnx_path: "models/yolov8s-seg.onnx"  # ONNX Runtime
```

Export a trained model to ONNX with:

```bash
python -m src.export_onnx --model-type yolo --weights models/yolov8s-seg.pt --output models/yolov8s-seg.onnx
python -m src.export_onnx --model-type tampernet --weights models/tamper_net.pt --output models/tamper_net.onnx
```

---

## Project Structure

```text
claim-guard-vision/
├── configs/
│   ├── model_config.yaml         # Image size, ELA quality, ONNX paths, thresholds
│   └── actuarial_weights.yaml    # Component weights w_i, gamma, tau_fraud, tau_cost
├── data/samples/                 # Sample inputs only; raw claim photos are git-ignored
├── src/
│   ├── forensics/
│   │   ├── ela.py                # ELA heatmap + statistical anomaly scorer
│   │   └── tamper_net.py         # Lightweight CNN / ONNX tamper-probability wrapper
│   ├── segmentation/
│   │   ├── damage_seg.py         # YOLOv8-seg / ONNX inference wrapper
│   │   └── geometry.py           # Mask area, r_i, IoU / Dice utilities
│   ├── actuarial/
│   │   └── risk_engine.py        # S_claim + rule-based audit triage
│   ├── synthetic/
│   │   └── gen_receipts.py       # Synthetic id_ID workshop receipts (Faker + Pillow)
│   └── export_onnx.py            # CLI: export PyTorch / YOLO weights to ONNX
├── app/dashboard.py              # Streamlit demo (Original | ELA | Segmentation + Decision)
├── tests/test_risk_engine.py     # Pytest: S_claim maths, edge cases, triage thresholds
├── requirements.txt              # CPU-friendly dependencies
└── README.md
```

## Configuration

| File | What to tune |
|---|---|
| `configs/model_config.yaml` | ELA quality/scale, TamperNet input size, segmentation backend/confidence, class names |
| `configs/actuarial_weights.yaml` | Per-component cost weights `w_i`, `gamma`, `tau_fraud`, `tau_cost`, per-component caps |

## Roadmap

- [ ] Train TamperNet on CASIA / Columbia splicing datasets; ship `models/tamper_net.onnx`
- [ ] Fine-tune YOLOv8-seg on annotated vehicle/property damage data
- [ ] Calibrate `w_i` / thresholds on historical claim + repair-invoice data
- [ ] Add OCR verification of synthetic receipts (Tesseract / TrOCR) against the JSON boxes
- [ ] FastAPI service wrapping the pipeline + audit-log persistence

## License

MIT — see `LICENSE`. Model weights, if any are added later, are distributed under their own licences and are never committed to this repository (see `.gitignore`).
