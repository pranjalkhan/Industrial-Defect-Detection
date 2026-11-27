"""
AI Industrial Quality Inspector - Interactive Dashboard

A modern, Figma-inspired industrial QC dashboard with:
- Live single-image inspection (upload or MVTec AD test sample picker)
- Interactive Plotly anomaly score gauge & threshold/opacity controls
- Multi-view defect localization (Original, Reconstructed, Heatmap, Overlay)
- Batch inspection across multiple images or MVTec test categories + CSV export
- Interactive 3-model benchmark comparison (Autoencoder vs ResNet18 vs PatchCore)
- Session inspection log & score trend analytics
"""

import os
import sys
import io
import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="GaugeAI · Industrial Defect Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS (Modern Industrial Control-Room UI)
# ============================================================
st.markdown(
    """
<style>
/* Global polish */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Header banner */
.hero-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0C4A6E 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 14px;
    padding: 1.25rem 1.6rem;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: #94A3B8;
    margin: 0.2rem 0 0 0;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #34D399;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* KPI Cards */
.kpi-card {
    background: #111827;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.kpi-card:hover {
    border-color: rgba(56, 189, 248, 0.45);
    transform: translateY(-1px);
}
.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 0.25rem;
}
.kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
}

/* Verdict Badges */
.verdict-pass {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(5, 150, 105, 0.08));
    border: 1px solid rgba(52, 211, 153, 0.5);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #34D399;
    font-weight: 700;
    font-size: 1.15rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.verdict-fail {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.20), rgba(185, 28, 28, 0.08));
    border: 1px solid rgba(248, 113, 113, 0.55);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #F87171;
    font-weight: 700;
    font-size: 1.15rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #111827;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.14);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Session State Initialization
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

COLORMAPS = {
    "JET (Classic)": cv2.COLORMAP_JET,
    "TURBO (High Contrast)": cv2.COLORMAP_TURBO,
    "INFERNO (Thermal)": cv2.COLORMAP_INFERNO,
    "HOT": cv2.COLORMAP_HOT,
}


# ============================================================
# Helper Functions
# ============================================================
@st.cache_resource
def load_local_detector(category: str, model_type: str):
    """Load and cache the detector + calibrated threshold for (category, model_type)."""
    from src.inference import get_detector

    return get_detector(category=category, model_type=model_type)


def run_local_inspection(image: Image.Image, category: str, model_type: str, custom_threshold: float, alpha: float, cmap_code: int):
    """Run inspection locally with live-adjustable threshold, opacity, and colormap."""
    from src.preprocessing import create_heatmap_overlay

    detector = load_local_detector(category, model_type)
    original_threshold = detector.threshold
    detector.threshold = custom_threshold
    result = detector.detect(image, return_visuals=True)
    detector.threshold = original_threshold

    anomaly_map = result["anomaly_map"]
    original_np = result["original"]
    overlay = create_heatmap_overlay(original_np, anomaly_map, alpha=alpha, colormap=cmap_code)

    if anomaly_map.max() > anomaly_map.min():
        norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        norm_map = np.zeros_like(anomaly_map)
    heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cmap_code)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    result["overlay"] = overlay
    result["heatmap_color"] = heatmap_color
    return result


def run_api_inspection(image: Image.Image, filename: str, category: str, model_type: str, api_url: str, custom_threshold: float):
    """Run inspection through the FastAPI endpoint."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    response = requests.post(
        f"{api_url}/predict",
        params={"category": category, "model_type": model_type},
        files={"file": (filename, buf.getvalue(), "image/png")},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"API Error {response.status_code}: {response.text}")
    data = response.json()
    score = data["anomaly_score"]
    data["threshold"] = custom_threshold
    data["prediction"] = "DEFECTIVE" if score > custom_threshold else "NORMAL"
    return data


def make_score_gauge(score: float, threshold: float) -> go.Figure:
    """Create an interactive Plotly gauge for the anomaly score."""
    max_val = max(threshold * 2.0, score * 1.25, 0.25)
    is_defect = score > threshold
    bar_color = "#EF4444" if is_defect else "#10B981"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={"valueformat": ".4f", "font": {"size": 30, "color": "#F8FAFC"}},
            delta={
                "reference": threshold,
                "valueformat": ".4f",
                "increasing": {"color": "#F87171"},
                "decreasing": {"color": "#34D399"},
            },
            title={"text": "Anomaly Score vs Threshold", "font": {"size": 14, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#64748B"},
                "bar": {"color": bar_color},
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, threshold], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [threshold, max_val], "color": "rgba(239, 68, 68, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#38BDF8", "width": 3},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC"},
    )
    return fig


def get_mvtec_samples(category: str = "bottle"):
    """List available MVTec AD test images grouped by defect type for a category."""
    test_root = PROJECT_ROOT / "data" / "mvtec_anomaly_detection" / category / "test"
    samples = {}
    if test_root.exists():
        for category_dir in sorted(test_root.iterdir()):
            if category_dir.is_dir():
                imgs = sorted([p for p in category_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
                if imgs:
                    samples[category_dir.name] = imgs
    return samples


# ============================================================
# Sidebar Controls
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Inspection Controls")

    from src.inference import list_available_categories, get_default_threshold

    available_categories = list_available_categories() or ["bottle"]
    selected_category = st.selectbox(
        "Product Category",
        available_categories,
        format_func=lambda x: x.replace("_", " ").title(),
        help="Select which trained product model to use for inspection.",
    )

    MODEL_OPTIONS = {
        "PatchCore (Best · SOTA)": "patchcore",
        "ResNet18 Features": "resnet",
        "ConvAutoencoder (Baseline)": "autoencoder",
    }
    model_label = st.selectbox(
        "Model Architecture",
        list(MODEL_OPTIONS.keys()),
        index=0,
        help="PatchCore achieves 0.95–1.00 AUROC across all categories.",
    )
    selected_model_type = MODEL_OPTIONS[model_label]

    inference_mode = st.radio(
        "Inference Engine",
        ["Direct (Local PyTorch)", "FastAPI Server"],
        help="Direct mode runs PyTorch locally with live colormap/opacity controls.",
    )
    api_url = "http://127.0.0.1:8000"
    if inference_mode == "FastAPI Server":
        api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000")

    st.divider()

    # Load default threshold for (selected_category, selected_model_type)
    loaded_thresh = get_default_threshold(selected_category, selected_model_type)
    if selected_model_type == "patchcore":
        t_min, t_max, t_step = 0.5, 12.0, 0.05
        default_thresh = loaded_thresh if loaded_thresh is not None else 3.50
    elif selected_model_type == "resnet":
        t_min, t_max, t_step = 0.5, 15.0, 0.05
        default_thresh = loaded_thresh if loaded_thresh is not None else 3.00
    else:
        t_min, t_max, t_step = 0.01, 0.40, 0.005
        default_thresh = loaded_thresh if loaded_thresh is not None else 0.15

    default_thresh = float(min(max(default_thresh, t_min), t_max))

    st.markdown("### 🎚️ Threshold & Heatmap")
    custom_threshold = st.slider(
        "Decision Threshold",
        min_value=t_min,
        max_value=t_max,
        value=float(round(default_thresh, 4)),
        step=t_step,
        format="%.4f",
        key=f"thresh_{selected_category}_{selected_model_type}",
        help="Scores above this threshold are flagged as DEFECTIVE.",
    )
    overlay_alpha = st.slider(
        "Heatmap Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
    )
    cmap_label = st.selectbox("Heatmap Colormap", list(COLORMAPS.keys()), index=1)
    cmap_code = COLORMAPS[cmap_label]

    st.divider()

    # Hardware & Session Summary
    st.markdown("### 🖥️ System Status")
    try:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        st.caption(f"**PyTorch:** `{torch.__version__}`")
        st.caption(f"**Compute:** `{gpu_name}`")
    except Exception:
        st.caption("PyTorch unavailable")

    total = len(st.session_state.history)
    normal_cnt = sum(1 for h in st.session_state.history if h["prediction"] == "NORMAL")
    defect_cnt = total - normal_cnt
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Pass", normal_cnt)
    c3.metric("Fail", defect_cnt)

    if st.button("🗑️ Clear Session Log", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ============================================================
# Top Hero Header
# ============================================================
st.markdown(
    f"""
<div class="hero-header">
  <div>
    <p class="hero-title">🔍 GaugeAI · Industrial Defect Inspection</p>
    <p class="hero-subtitle">Real-time anomaly detection, pixel-level localization &amp; multi-model benchmarking (MVTec AD · <b>{selected_category.replace('_', ' ').title()}</b>)</p>
  </div>
  <div class="status-pill">● SYSTEM ONLINE</div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Main Navigation Tabs
# ============================================================
tab_inspect, tab_batch, tab_benchmarks, tab_history = st.tabs(
    ["🔍 Live Inspection", "📦 Batch Inspection", "📊 Model Benchmarks", "📋 Inspection Log"]
)

# ------------------------------------------------------------
# TAB 1: Live Inspection
# ------------------------------------------------------------
with tab_inspect:
    source_col, preview_col = st.columns([1.1, 1.9], gap="large")

    mvtec_samples = get_mvtec_samples(selected_category)
    selected_image = None
    selected_name = None
    ground_truth_label = None

    with source_col:
        st.markdown("#### 1. Select Product Image")
        input_source = st.radio(
            "Image Source",
            ["Sample from MVTec Test Set", "Upload Custom Image"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if input_source == "Sample from MVTec Test Set" and mvtec_samples:
            cat_choice = st.selectbox("Test Category", list(mvtec_samples.keys()), index=1 if len(mvtec_samples) > 1 else 0)
            img_paths = mvtec_samples[cat_choice]
            img_names = [p.name for p in img_paths]
            img_choice = st.selectbox("Sample Image", img_names)
            chosen_path = next(p for p in img_paths if p.name == img_choice)
            selected_image = Image.open(chosen_path).convert("RGB")
            selected_name = f"{cat_choice}/{img_choice}"
            ground_truth_label = "NORMAL" if cat_choice == "good" else f"DEFECTIVE ({cat_choice})"
        else:
            uploaded = st.file_uploader(
                "Drop a product image (PNG / JPG)",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded:
                selected_image = Image.open(uploaded).convert("RGB")
                selected_name = uploaded.name

        if selected_image is not None:
            st.image(selected_image, caption=f"Input: {selected_name}", use_container_width=True)
            if ground_truth_label:
                st.caption(f"Ground Truth: **{ground_truth_label}**")

            inspect_clicked = st.button("🚀 Run AI Inspection", type="primary", use_container_width=True)
        else:
            st.info("Upload an image or pick a sample from the MVTec AD test set to begin.")
            inspect_clicked = False

    with preview_col:
        if selected_image is not None and inspect_clicked:
            with st.spinner("Running neural inspection & localization..."):
                try:
                    if inference_mode == "Direct (Local PyTorch)":
                        res = run_local_inspection(selected_image, selected_category, selected_model_type, custom_threshold, overlay_alpha, cmap_code)
                    else:
                        res = run_api_inspection(selected_image, selected_name, selected_category, selected_model_type, api_url, custom_threshold)
                except Exception as exc:
                    st.error(f"Inspection failed: {exc}")
                    res = None

            if res:
                is_pass = res["prediction"] == "NORMAL"
                badge_cls = "verdict-pass" if is_pass else "verdict-fail"
                badge_icon = "✅ PASS · NORMAL" if is_pass else "🚨 FAIL · DEFECTIVE"
                ratio = res["anomaly_score"] / custom_threshold if custom_threshold > 0 else 0.0

                st.markdown(
                    f'<div class="{badge_cls}"><span>{badge_icon}</span><span>{ratio:.2f}× threshold</span></div>',
                    unsafe_allow_html=True,
                )

                # Gauge + KPI row
                g_col, k_col = st.columns([1.4, 1.0])
                with g_col:
                    st.plotly_chart(make_score_gauge(res["anomaly_score"], custom_threshold), use_container_width=True)
                with k_col:
                    st.markdown(
                        f"""
                        <div class="kpi-card" style="margin-top: 1rem;">
                          <div class="kpi-label">Anomaly Score</div>
                          <div class="kpi-value">{res['anomaly_score']:.6f}</div>
                        </div>
                        <div class="kpi-card" style="margin-top: 0.75rem;">
                          <div class="kpi-label">Active Threshold</div>
                          <div class="kpi-value">{custom_threshold:.4f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Visual localization panels
                st.markdown("#### 2. Defect Localization Views")
                if inference_mode == "Direct (Local PyTorch)":
                    if "reconstructed" in res:
                        v1, v2, v3, v4 = st.columns(4)
                        v1.image(res["original"], caption="Original", use_container_width=True)
                        v2.image(res["reconstructed"], caption="Reconstructed", use_container_width=True)
                        v3.image(res["heatmap_color"], caption="Anomaly Map", use_container_width=True)
                        v4.image(res["overlay"], caption="Localization", use_container_width=True)
                    else:
                        v1, v2, v3 = st.columns(3)
                        v1.image(res["original"], caption="Original", use_container_width=True)
                        v2.image(res["heatmap_color"], caption="Anomaly Map", use_container_width=True)
                        v3.image(res["overlay"], caption="Defect Localization", use_container_width=True)
                else:
                    v1, v2 = st.columns(2)
                    if res.get("heatmap_base64"):
                        hm_img = Image.open(io.BytesIO(base64.b64decode(res["heatmap_base64"])))
                        v1.image(hm_img, caption="Anomaly Map", use_container_width=True)
                    if res.get("overlay_base64"):
                        ov_img = Image.open(io.BytesIO(base64.b64decode(res["overlay_base64"])))
                        v2.image(ov_img, caption="Defect Localization", use_container_width=True)

                # Log to session history
                st.session_state.history.insert(
                    0,
                    {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "sample": selected_name,
                        "prediction": res["prediction"],
                        "anomaly_score": round(float(res["anomaly_score"]), 6),
                        "threshold": round(float(custom_threshold), 4),
                    },
                )
        else:
            st.markdown(
                """
                <div class="kpi-card" style="text-align:center; padding: 3rem 1.5rem; margin-top: 2rem;">
                  <div style="font-size: 2.5rem;">🔬</div>
                  <h4 style="margin: 0.5rem 0; color: #F8FAFC;">Ready for Inspection</h4>
                  <p style="color: #94A3B8; margin: 0;">Select an image on the left and click <b>Run AI Inspection</b> to view the anomaly score gauge and pixel-level localization heatmap.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------
# TAB 2: Batch Inspection
# ------------------------------------------------------------
with tab_batch:
    st.markdown("#### 📦 Batch Quality Inspection")
    st.caption("Inspect multiple uploaded images or an entire MVTec AD test category at once.")

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        batch_source = st.radio("Batch Source", ["MVTec Test Category", "Upload Multiple Files"])
        batch_files = []
        if batch_source == "MVTec Test Category" and mvtec_samples:
            b_cat = st.selectbox("Category to Inspect", list(mvtec_samples.keys()), key="batch_cat")
            batch_paths = mvtec_samples[b_cat]
            st.write(f"Found **{len(batch_paths)}** images in `test/{b_cat}`.")
            run_batch = st.button("⚡ Run Batch Inspection", type="primary", use_container_width=True)
        else:
            uploaded_many = st.file_uploader(
                "Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
            )
            batch_files = uploaded_many or []
            run_batch = st.button(
                f"⚡ Inspect {len(batch_files)} Images",
                type="primary",
                use_container_width=True,
                disabled=len(batch_files) == 0,
            )

    with b_col2:
        if run_batch:
            rows = []
            progress = st.progress(0.0, text="Starting batch inspection...")
            items = (
                [(p.name, Image.open(p).convert("RGB")) for p in mvtec_samples[b_cat]]
                if batch_source == "MVTec Test Category"
                else [(f.name, Image.open(f).convert("RGB")) for f in batch_files]
            )
            for idx, (fname, img) in enumerate(items):
                res = run_local_inspection(img, selected_category, selected_model_type, custom_threshold, overlay_alpha, cmap_code)
                rows.append(
                    {
                        "Image": fname,
                        "Prediction": res["prediction"],
                        "Anomaly Score": round(float(res["anomaly_score"]), 6),
                        "Threshold": custom_threshold,
                    }
                )
                progress.progress((idx + 1) / len(items), text=f"Inspected {idx + 1}/{len(items)}: {fname}")

            df_batch = pd.DataFrame(rows)
            n_total = len(df_batch)
            n_fail = int((df_batch["Prediction"] == "DEFECTIVE").sum())
            n_pass = n_total - n_fail

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Inspected", n_total)
            m2.metric("Normal (PASS)", n_pass)
            m3.metric("Defective (FAIL)", n_fail)
            m4.metric("Pass Rate", f"{(n_pass / max(n_total, 1)) * 100:.1f}%")

            fig_bar = px.bar(
                df_batch,
                x="Image",
                y="Anomaly Score",
                color="Prediction",
                color_discrete_map={"NORMAL": "#10B981", "DEFECTIVE": "#EF4444"},
                title="Batch Anomaly Scores",
            )
            fig_bar.add_hline(y=custom_threshold, line_dash="dash", line_color="#38BDF8", annotation_text="Threshold")
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.6)", font_color="#F8FAFC")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(df_batch, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download Batch Report (CSV)",
                data=df_batch.to_csv(index=False).encode("utf-8"),
                file_name=f"batch_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ------------------------------------------------------------
# TAB 3: Model Benchmarks
# ------------------------------------------------------------
with tab_benchmarks:
    st.markdown(f"#### 📊 Multi-Model Benchmark Comparison (MVTec AD · {selected_category.replace('_', ' ').title()})")
    cat_comp_path = PROJECT_ROOT / "artifacts" / selected_category / "model_comparison.json"
    legacy_comp_path = PROJECT_ROOT / "artifacts" / "model_comparison.json"
    comp_path = cat_comp_path if cat_comp_path.exists() else legacy_comp_path

    if comp_path.exists():
        with open(comp_path, "r") as f:
            comp_data = json.load(f)

        df_comp = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Image AUROC": m.get("image_auroc", 0.0),
                    "Pixel AUROC": m.get("pixel_auroc", 0.0),
                    "Image F1": m.get("image_f1", 0.0),
                    "Inference (ms/img)": m.get("inference_ms", 0.0),
                    "Artifact Size (MB)": m.get("model_size_mb", 0.0),
                }
                for name, m in comp_data.items()
            ]
        )

        # Top KPI Cards for Best Model
        best_row = df_comp.sort_values("Pixel AUROC", ascending=False).iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Top Model", best_row["Model"])
        k2.metric("Best Image AUROC", f"{df_comp['Image AUROC'].max():.4f}")
        k3.metric("Best Pixel AUROC", f"{df_comp['Pixel AUROC'].max():.4f}")
        k4.metric("Best Image F1", f"{df_comp['Image F1'].max():.4f}")

        c_left, c_right = st.columns(2)
        with c_left:
            df_melt = df_comp.melt(
                id_vars="Model",
                value_vars=["Image AUROC", "Pixel AUROC", "Image F1"],
                var_name="Metric",
                value_name="Score",
            )
            fig_acc = px.bar(
                df_melt,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Detection & Localization Accuracy",
                color_discrete_sequence=["#38BDF8", "#34D399", "#A78BFA"],
            )
            fig_acc.update_layout(
                yaxis_range=[0, 1.08],
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font_color="#F8FAFC",
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with c_right:
            fig_speed = px.scatter(
                df_comp,
                x="Inference (ms/img)",
                y="Pixel AUROC",
                size="Artifact Size (MB)",
                color="Model",
                text="Model",
                title="Speed vs Accuracy Trade-off (bubble = size MB)",
                color_discrete_sequence=["#F59E0B", "#38BDF8", "#10B981"],
            )
            fig_speed.update_traces(textposition="top center")
            fig_speed.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font_color="#F8FAFC",
            )
            st.plotly_chart(fig_speed, use_container_width=True)

        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        cat_eval_plot = PROJECT_ROOT / "artifacts" / selected_category / "evaluation" / "evaluation_plots.png"
        legacy_eval_plot = PROJECT_ROOT / "artifacts" / "evaluation" / "evaluation_plots.png"
        eval_plot = cat_eval_plot if cat_eval_plot.exists() else legacy_eval_plot
        if eval_plot.exists():
            with st.expander("📈 Autoencoder ROC, Score Distribution & Confusion Matrix"):
                st.image(str(eval_plot), use_container_width=True)

    all_comp_path = PROJECT_ROOT / "artifacts" / "all_categories_comparison.json"
    if all_comp_path.exists():
        st.divider()
        st.markdown("#### 🌐 All Categories Summary (PatchCore / ResNet / Autoencoder)")
        with open(all_comp_path, "r") as f:
            all_data = json.load(f)
        summary_rows = []
        for cat_name, models_dict in all_data.items():
            for m_name, m_vals in models_dict.items():
                summary_rows.append({
                    "Category": cat_name.replace("_", " ").title(),
                    "Model": m_name,
                    "Image AUROC": round(m_vals.get("image_auroc", 0.0), 4),
                    "Pixel AUROC": round(m_vals.get("pixel_auroc", 0.0), 4),
                    "Image F1": round(m_vals.get("image_f1", 0.0), 4),
                    "Inference (ms)": m_vals.get("inference_ms", 0.0),
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python run_pipeline.py --compare` to generate benchmark data.")


# ------------------------------------------------------------
# TAB 4: Session Inspection Log
# ------------------------------------------------------------
with tab_history:
    st.markdown("#### 📋 Live Session Inspection Log")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if len(df_hist) > 1:
            fig_trend = px.line(
                df_hist.iloc[::-1],
                x="timestamp",
                y="anomaly_score",
                markers=True,
                title="Anomaly Score Trend Across Inspections",
                color_discrete_sequence=["#38BDF8"],
            )
            fig_trend.add_hline(y=custom_threshold, line_dash="dash", line_color="#EF4444", annotation_text="Threshold")
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17,24,39,0.6)",
                font_color="#F8FAFC",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.download_button(
            "⬇️ Export Session Log (CSV)",
            data=df_hist.to_csv(index=False).encode("utf-8"),
            file_name="inspection_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No inspections logged in this session yet. Run an inspection in the **Live Inspection** tab.")
