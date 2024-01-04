"""Streamlit app for local document tampering inspection with ELA preprocessing, MC Dropout uncertainty, patch heatmaps, OCR, and rule-based fusion.
Model weights and architecture remain unchanged; SageMaker is not used here."""
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

from ela import convert_to_ela_image

# Cache the CNN once for local inference.
_model: Any = None


def load_model() -> Any:
    """Load Keras model from disk with caching."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(os.path.join("model", "1"))
    return _model


def ela_from_bytes(image_bytes: bytes, quality: int = 90) -> Tuple[Image.Image, Image.Image, str]:
    """Process uploaded bytes into original, ELA, and temp path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(image_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()
    ela_img = convert_to_ela_image(tmp_path, quality).convert("RGB")
    orig_img = Image.open(tmp_path).convert("RGB")
    return orig_img, ela_img, tmp_path


def to_input_tensor(ela_image: Image.Image) -> np.ndarray:
    """Convert ELA image to normalized (1,128,128,3) tensor."""
    arr = np.array(ela_image.resize((128, 128))).astype(np.float32) / 255.0
    return arr.reshape(1, 128, 128, 3)


def predict_tamper_prob(x: np.ndarray) -> float:
    """Single forward pass to get tamper probability."""
    model = load_model()
    preds = model.predict(x, verbose=0)
    return float(preds[0][1])


def mc_dropout_predict(x: np.ndarray, samples: int = 20, uncertainty_threshold: float = 0.15) -> Tuple[float, float, bool]:
    """MC Dropout: multiple stochastic forward passes."""
    model = load_model()
    probs = []
    for _ in range(samples):
        preds = model(x, training=True).numpy()  # enable dropout
        probs.append(float(preds[0][1]))
    mean_prob = float(np.mean(probs))
    uncertainty = float(np.std(probs))
    low_conf = uncertainty >= uncertainty_threshold
    return mean_prob, uncertainty, low_conf


def compute_patch_heatmap(ela_image: Image.Image, patch_size: int = 128, stride: int = 64) -> np.ndarray:
    """Overlapping patch scores to build tamper probability heatmap."""
    model = load_model()
    w, h = ela_image.size
    xs = list(range(0, max(w - patch_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(h - patch_size, 0) + 1, stride)) or [0]
    heatmap_grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            patch = ela_image.crop((x, y, min(x + patch_size, w), min(y + patch_size, h)))
            patch_arr = to_input_tensor(patch)
            preds = model.predict(patch_arr, verbose=0)
            heatmap_grid[yi, xi] = float(preds[0][1])
    heatmap_resized = cv2.resize(heatmap_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / heatmap_resized.max()
    return heatmap_resized


def overlay_heatmap(base_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay heatmap on original image."""
    base = base_image.convert("RGB")
    heat_resized = cv2.resize(heatmap, base.size)
    # Convert to uint8 for colormap: applyColorMap requires single-channel uint8 input
    # Clip to [0,1] first, then scale to 0-255 range, then cast to uint8
    heat_clipped = np.clip(heat_resized, 0.0, 1.0).astype(np.float64)  # type: ignore[arg-type]
    heat_uint8 = (heat_clipped * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # type: ignore[arg-type]
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    # Type: ignore because np.ndarray multiplication with float is valid for OpenCV blending
    overlay = np.array(base) * (1 - alpha) + heat_color * alpha  # type: ignore[operator]
    return Image.fromarray(np.uint8(overlay))


def run_ocr(image: Image.Image, conf_threshold: int = 40) -> list:
    """Extract text and boxes from image via Tesseract."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i].strip() if data["text"][i] else ""
        try:
            conf = int(float(data["conf"][i]))
        except Exception:
            conf = -1
        if text and conf >= conf_threshold:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            results.append({"text": text, "conf": conf, "bbox": [int(x), int(y), int(x + w), int(y + h)]})
    return results


def draw_ocr_boxes(image: Image.Image, ocr_results: list) -> Image.Image:
    """Draw red boxes around OCR detections."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for item in ocr_results:
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return img


def _ocr_box_stats(ocr_results: list) -> dict:
    """Compute stats for OCR boxes."""
    if not ocr_results:
        return {"widths": [], "heights": [], "spacings": []}
    widths = [b["bbox"][2] - b["bbox"][0] for b in ocr_results]
    heights = [b["bbox"][3] - b["bbox"][1] for b in ocr_results]
    sorted_boxes = sorted(ocr_results, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    spacings = []
    for i in range(1, len(sorted_boxes)):
        prev = sorted_boxes[i - 1]["bbox"]
        curr = sorted_boxes[i]["bbox"]
        if heights and abs(curr[1] - prev[1]) < 0.5 * heights[i - 1]:
            spacings.append(curr[0] - prev[2])
        else:
            spacings.append(curr[1] - prev[3])
    return {"widths": widths, "heights": heights, "spacings": spacings}


def _flag_ocr_anomalies(ocr_results: list) -> list:
    """Flag OCR text consistency and font/spacing anomalies."""
    stats = _ocr_box_stats(ocr_results)
    anomalies = []
    if not ocr_results:
        anomalies.append("no_ocr_text_detected")
        return anomalies

    def cv(values):
        if not values:
            return 0.0
        mean = float(np.mean(values))
        std = float(np.std(values))
        return std / mean if mean > 0 else 0.0

    if cv(stats["heights"]) > 0.35:
        anomalies.append("font_height_variation_high")
    if cv(stats["widths"]) > 0.60:
        anomalies.append("font_width_variation_high")
    if cv(stats["spacings"]) > 1.0:
        anomalies.append("spacing_inconsistent")
    lengths = [len(r["text"]) for r in ocr_results]
    if lengths and np.median(lengths) <= 2 and np.mean(lengths) <= 3:
        anomalies.append("text_fragments_only")
    return anomalies


def correlate_heatmap_with_ocr(patch_heatmap: np.ndarray, ocr_results: list) -> float:
    """Correlate tamper heatmap with OCR regions."""
    if patch_heatmap is None or patch_heatmap.size == 0 or not ocr_results:
        return 0.0
    h, w = patch_heatmap.shape
    scores = []
    for item in ocr_results:
        x1, y1, x2, y2 = item["bbox"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w - 1, x2), min(h - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = patch_heatmap[y1c:y2c, x1c:x2c]
        if crop.size > 0:
            scores.append(float(np.mean(crop)))
    return float(np.max(scores)) if scores else 0.0


def fuse_signals(tamper_prob: float, mc_mean: float, mc_uncertainty: float, patch_heatmap: np.ndarray, ocr_results: list) -> Tuple[float, dict]:
    """Fuse OCR, heatmap, and model signals into risk score."""
    anomalies = _flag_ocr_anomalies(ocr_results)
    heat_ocr_overlap = correlate_heatmap_with_ocr(patch_heatmap, ocr_results)
    flags = []
    if anomalies:
        flags.extend(anomalies)
    if heat_ocr_overlap > 0.35:
        flags.append("ocr_region_aligned_with_heatmap")
    risk_parts = [tamper_prob, mc_mean, heat_ocr_overlap]
    risk = float(np.mean(risk_parts))
    if anomalies:
        risk += 0.1
    risk += min(mc_uncertainty, 0.3)
    risk = min(risk, 1.0)
    explanation = {
        "tamper_prob_endpoint": tamper_prob,
        "mc_dropout_mean": mc_mean,
        "mc_dropout_uncertainty": mc_uncertainty,
        "heat_ocr_overlap": heat_ocr_overlap,
        "anomalies": flags,
        "risk_score": risk,
    }
    return risk, explanation


# Streamlit UI
st.set_page_config(page_title="Document Tampering Inspector", layout="wide")
st.title("Document Tampering Inspector (Local)")
st.caption("Uses ELA preprocessing, CNN classifier, MC Dropout, patch heatmaps, OCR, and rule-based fusion. Model weights remain unchanged; no SageMaker required.")

uploaded = st.file_uploader("Upload a document image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    bytes_data = uploaded.read()
    orig_img, ela_img, tmp_path = ela_from_bytes(bytes_data)
    try:
        x = to_input_tensor(ela_img)
        tamper_prob = predict_tamper_prob(x)
        mc_mean, mc_uncertainty, low_conf = mc_dropout_predict(x)
        patch_heatmap = compute_patch_heatmap(ela_img)
        heat_overlay = overlay_heatmap(orig_img, patch_heatmap)
        ocr_results = run_ocr(orig_img)
        ocr_overlay = draw_ocr_boxes(orig_img, ocr_results)
        risk_score, explanation = fuse_signals(tamper_prob, mc_mean, mc_uncertainty, patch_heatmap, ocr_results)
    finally:
        os.unlink(tmp_path)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(orig_img, use_column_width=True)
    with col2:
        st.subheader("Tamper Heatmap")
        st.image(heat_overlay, use_column_width=True)
    with col3:
        st.subheader("OCR Boxes")
        st.image(ocr_overlay, use_column_width=True)

    st.markdown("---")
    st.subheader("Scores")
    st.metric("Tamper probability", f"{tamper_prob:.3f}")
    st.metric("MC Dropout mean", f"{mc_mean:.3f}")
    st.metric("Uncertainty (std)", f"{mc_uncertainty:.3f}")
    st.metric("Risk score", f"{risk_score:.3f}")
    st.write("Low confidence flag:", "manual review required" if low_conf else "model confidence acceptable")

    st.markdown("---")
    st.subheader("OCR Results")
    if ocr_results:
        st.dataframe(ocr_results)
    else:
        st.info("No OCR text detected above confidence threshold.")

    st.markdown("---")
    st.subheader("Fusion Explanation (JSON)")
    st.code(json.dumps(explanation, indent=2))
else:
    st.info("Upload an image to begin.")
