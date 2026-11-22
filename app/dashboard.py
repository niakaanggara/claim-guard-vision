"""Interactive 3-panel Streamlit demo for claim-guard-vision.

Panels: (1) Original Image, (2) ELA Heatmap, (3) Segmentation + Audit Decision.

Run with::

    streamlit run app/dashboard.py

The demo works without any model weights: ELA and the actuarial engine
run on CPU, and segmentation degrades gracefully to an ELA-only view
with a clear message when no YOLO / ONNX weights are configured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image

# Allow `streamlit run app/dashboard.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.actuarial.risk_engine import DamageComponent, RiskEngine, load_actuarial_config  # noqa: E402
from src.forensics.ela import compute_ela  # noqa: E402
from src.segmentation.geometry import mask_area_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def _load_configs():
    with open(ROOT / "configs" / "model_config.yaml", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    actuarial_cfg = load_actuarial_config(ROOT / "configs" / "actuarial_weights.yaml")
    return model_cfg, actuarial_cfg


def _overlay_masks(image: Image.Image, result) -> Image.Image:
    """Draw segmentation masks translucently over ``image`` (Pillow only)."""
    import numpy as np

    base = np.asarray(image.convert("RGB")).copy()
    colours = [(255, 0, 0), (0, 180, 0), (0, 120, 255), (255, 165, 0), (180, 0, 180)]
    for i, mask in enumerate(result.masks):
        colour = np.array(colours[i % len(colours)], dtype=np.uint8)
        m = np.asarray(mask) > 0
        base[m] = (0.6 * base[m] + 0.4 * colour).astype(np.uint8)
    return Image.fromarray(base)


def main() -> None:
    st.set_page_config(page_title="Claim Guard Vision", layout="wide")
    st.title("🛡️ Claim Guard Vision")
    st.caption("Forensic image tampering detection & actuarial claim severity estimation")

    model_cfg, actuarial_cfg = _load_configs()
    engine = RiskEngine(actuarial_cfg)

    with st.sidebar:
        st.header("Thresholds")
        tau_fraud = st.slider("τ_fraud (tamper)", 0.0, 1.0, float(actuarial_cfg.tau_fraud), 0.05)
        tau_cost = st.slider("τ_cost (severity)", 0.0, 1.0, float(actuarial_cfg.tau_cost), 0.05)
        engine.config.tau_fraud = tau_fraud
        engine.config.tau_cost = tau_cost
        st.header("ELA")
        ela_quality = st.slider("JPEG quality", 50, 95, int(model_cfg["ela"]["quality"]), 5)
        st.caption("Upload a claim photo (JPG/PNG). No image is stored; it is processed in memory.")

    uploaded = st.file_uploader("Claim photo", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a claim photo to run the pipeline. Try a photo of vehicle or property damage.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    ela = compute_ela(image, quality=ela_quality, scale=float(model_cfg["ela"]["scale"]))

    # Segmentation (best effort — the demo must not crash without weights).
    seg_result = None
    seg_error = ""
    try:
        from src.segmentation.damage_seg import DamageSegmenter

        seg_cfg = model_cfg["segmentation"]
        segmenter = DamageSegmenter(
            weights=ROOT / seg_cfg["weights"],
            onnx_path=ROOT / seg_cfg["onnx_path"],
            backend=seg_cfg.get("backend", "auto"),
            confidence=float(seg_cfg["confidence"]),
        )
        seg_result = segmenter.predict(image)
    except Exception as exc:  # noqa: BLE001 - show the reason in the UI
        seg_error = str(exc)

    # Actuarial assessment: r_i from masks, or 0 when segmentation is unavailable.
    components: list[DamageComponent] = []
    if seg_result is not None and len(seg_result) > 0:
        ref_area = float(image.width * image.height)
        for mask, name in zip(seg_result.masks, seg_result.class_names):
            components.append(DamageComponent(name, mask_area_ratio(mask, ref_area)))
    assessment = engine.assess(components, p_tamper=ela.p_tamper)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1 · Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("2 · ELA Heatmap")
        st.image(ela.heatmap, use_container_width=True)
        st.metric("P_tamper (ELA)", f"{ela.p_tamper:.3f}")
        st.write(f"Mean error {ela.mean_error:.2f} · p95 {ela.p95_error:.2f} · "
                 f"{ela.fraction_above_threshold:.1%} pixels anomalous")
    with col3:
        st.subheader("3 · Segmentation + Decision")
        if seg_result is not None and len(seg_result) > 0:
            st.image(_overlay_masks(image, seg_result), use_container_width=True)
            st.write(f"{len(seg_result)} damage instance(s) — backend: {seg_result.backend}")
            for name, score, ratio in zip(seg_result.class_names, seg_result.scores, [c.area_ratio for c in components]):
                st.write(f"• {name}: conf {score:.2f}, r = {ratio:.3f}")
        else:
            st.warning(seg_error or "No damage instances detected.")
            st.image(image, use_container_width=True)

    st.divider()
    decision = assessment.decision.value
    if "REJECT" in decision:
        st.error(f"**Decision: {decision}**")
    elif "MANUAL" in decision:
        st.warning(f"**Decision: {decision}**")
    else:
        st.success(f"**Decision: {decision}**")
    st.metric("Severity index S_claim", f"{assessment.severity:.3f}")
    for reason in assessment.reasons:
        st.write(reason)
    if assessment.contributions:
        st.json({k: round(v, 4) for k, v in assessment.contributions.items()})


if __name__ == "__main__":
    main()
