
from pathlib import Path
import tempfile
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "screw_synthetic_experiment"
    / "Patchcore"
    / "MVTecAD"
    / "screw"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

# Actual test evaluation results
METRICS = {
    "Image AUROC": 0.955114,
    "Image AP": 0.984101,
    "Image F1": 0.875576,
    "Image Precision": 0.969388,
    "Image Recall": 0.798319,
    "Image FNR": 0.201681,
    "Pixel AUROC": 0.988939,
    "Pixel AP": 0.329947,
    "Pixel F1": 0.022767,
    "Pixel Precision": 0.724880,
    "Pixel Recall": 0.011565,
}

CONFUSION_MATRIX = np.array([
    [38, 3],
    [24, 95]
])


# --------------------------------------------------
# 2. CUSTOM DARK THEME
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263244;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, span {
    color: #cbd5e1;
}

.metric-card {
    background: linear-gradient(135deg, #172338, #111827);
    border: 1px solid #293b55;
    border-radius: 12px;
    padding: 18px;
    min-height: 115px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}

.section-card {
    background-color: #111c2e;
    border: 1px solid #293b55;
    border-radius: 12px;
    padding: 18px;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD PATCHCORE MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT}"
        )

    model = Patchcore()

    engine = Engine(
        default_root_dir=PROJECT_ROOT / "results" / "prototype"
    )

    return model, engine


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_heatmap(original_image, anomaly_map):
    original = np.array(original_image.convert("RGB"))

    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    anomaly_map = np.squeeze(anomaly_map)

    anomaly_map = np.nan_to_num(anomaly_map)

    low = anomaly_map.min()
    high = anomaly_map.max()

    if high > low:
        anomaly_map = (anomaly_map - low) / (high - low)
    else:
        anomaly_map = np.zeros_like(anomaly_map)

    heatmap = plt.get_cmap("jet")(anomaly_map)[..., :3]

    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap_image = Image.fromarray(heatmap).resize(
        original_image.size
    )

    heatmap_array = np.array(heatmap_image)

    overlay = (
        0.55 * original.astype(np.float32)
        + 0.45 * heatmap_array.astype(np.float32)
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def inspect_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / uploaded_file.name
        image.save(image_path)

        dataset = PredictDataset(
            path=temp_dir,
            image_size=(256, 256)
        )

        model, engine = load_model()

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=str(CHECKPOINT),
            return_predictions=True
        )

    if not predictions:
        raise RuntimeError("The model returned no predictions.")

    batch = predictions[0]

    score = float(batch.pred_score[0].item())
    label = int(batch.pred_label[0].item())

    anomaly_map = batch.anomaly_map[0].detach().cpu().numpy()

    overlay = create_heatmap(image, anomaly_map)

    return image, score, label, anomaly_map, overlay


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🔍 Inspection AI")

st.sidebar.caption("CNN-Based Manufacturing Defect Detection")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Live Inspection",
        "Model Performance",
        "Inspection History"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")
st.sidebar.write("**Model:** PatchCore")
st.sidebar.write("**Feature Extractor:** WideResNet50_2")
st.sidebar.write("**Dataset:** MVTec AD")
st.sidebar.write("**Category:** Screw")
st.sidebar.write("**Device:** CPU")

if CHECKPOINT.exists():
    st.sidebar.success("Model checkpoint available")
else:
    st.sidebar.error("Model checkpoint missing")


# --------------------------------------------------
# 6. HEADER
# --------------------------------------------------

st.title("Industrial Defect Detection System")

st.caption(
    "AI-powered visual inspection for manufacturing components"
)

st.divider()


# --------------------------------------------------
# 7. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.subheader("Dashboard Overview")

    st.write(
        "Monitor model performance and inspect manufacturing "
        "components for potential defects."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Image AUROC", "95.51%")

    with col2:
        metric_card("Image Precision", "96.94%")

    with col3:
        metric_card("Image Recall", "79.83%")

    with col4:
        metric_card("False Negative Rate", "20.17%")

    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown("### Detection Performance")

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        performance = pd.DataFrame({
            "Metric": ["AUROC", "Average Precision", "F1 Score"],
            "Value": [95.51, 98.41, 87.56]
        })

        st.bar_chart(
            performance.set_index("Metric"),
            y="Value"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### Prediction Summary")

        st.markdown(
            '<div class="section-card">',
            unsafe_allow_html=True
        )

        st.write("**Test images:** 160")
        st.write("**Normal images:** 41")
        st.write("**Defective images:** 119")
        st.write("**Correct normal predictions:** 38")
        st.write("**Correct defect predictions:** 95")
        st.write("**Missed defects:** 24")

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.info(
        "The system uses PatchCore anomaly detection trained "
        "using normal training images. Suspicious predictions "
        "should be reviewed by a human inspector."
    )


# --------------------------------------------------
# 8. LIVE INSPECTION PAGE
# --------------------------------------------------

elif page == "Live Inspection":

    st.subheader("Live Component Inspection")

    st.write(
        "Upload a screw image to identify potential anomalies."
    )

    uploaded_file = st.file_uploader(
        "Upload component image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:

        original = Image.open(uploaded_file).convert("RGB")

        st.markdown("### Uploaded Image")
        st.image(original, use_container_width=True)

        if st.button("Run Defect Inspection", use_container_width=True):

            try:
                with st.spinner("Analysing component..."):

                    image, score, label, anomaly_map, overlay = (
                        inspect_image(uploaded_file)
                    )

                result = (
                    "Suspicious Component"
                    if label == 1
                    else "Normal Component"
                )

                st.session_state.setdefault("history", [])

                st.session_state.history.insert(
                    0,
                    {
                        "Image": uploaded_file.name,
                        "Prediction": result,
                        "Anomaly Score": score,
                        "Time": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    }
                )

                st.divider()

                st.subheader("Inspection Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Anomaly Score", f"{score:.4f}")

                with col2:
                    if label == 1:
                        st.error("Suspicious Component")
                    else:
                        st.success("Normal Component")

                st.markdown("### Defect Localization")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Original Image")
                    st.image(image, use_container_width=True)

                with col2:
                    st.write("Anomaly Heatmap Overlay")
                    st.image(
                        overlay,
                        use_container_width=True
                    )

                st.warning(
                    "The heatmap is an anomaly visualization, "
                    "not a verified defect segmentation mask. "
                    "Please review suspicious or uncertain "
                    "components manually."
                )

            except Exception as error:
                st.error(f"Inspection failed: {error}")


# --------------------------------------------------
# 9. MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.subheader("Model Evaluation")

    st.caption(
        "Metrics below are from the completed test evaluation."
    )

    st.markdown("### Image-Level Metrics")

    columns = st.columns(3)

    image_metrics = [
        ("AUROC", "95.51%"),
        ("Average Precision", "98.41%"),
        ("F1 Score", "87.56%"),
        ("Precision", "96.94%"),
        ("Recall", "79.83%"),
        ("False Negative Rate", "20.17%")
    ]

    for index, (name, value) in enumerate(image_metrics):
        with columns[index % 3]:
            metric_card(name, value)

    st.write("")

    st.markdown("### Pixel-Level Metrics")

    columns = st.columns(3)

    pixel_metrics = [
        ("AUROC", "98.89%"),
        ("Average Precision", "32.99%"),
        ("F1 Score", "2.28%"),
        ("Precision", "72.49%"),
        ("Recall", "1.16%"),
    ]

    for index, (name, value) in enumerate(pixel_metrics):
        with columns[index % 3]:
            metric_card(name, value)

    st.divider()

    st.markdown("### Image-Level Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 4))

    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    ax.imshow(CONFUSION_MATRIX, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Defective"], color="white")
    ax.set_yticklabels(["Normal", "Defective"], color="white")

    ax.set_xlabel("Predicted Label", color="white")
    ax.set_ylabel("Actual Label", color="white")

    ax.set_title("Confusion Matrix", color="white")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                str(CONFUSION_MATRIX[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold"
            )

    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "Pixel-level F1 and recall are low at the evaluated "
        "threshold. High pixel AUROC alone does not establish "
        "accurate defect localization."
    )


# --------------------------------------------------
# 10. INSPECTION HISTORY PAGE
# --------------------------------------------------

elif page == "Inspection History":

    st.subheader("Inspection History")

    history = st.session_state.get("history", [])

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Inspection History",
            data=csv,
            file_name="inspection_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Inspection History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No inspections yet. Go to Live Inspection "
            "to upload and analyse a component."
        )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "CNN-Based Defect Detection in Manufacturing Processes | "
    "PatchCore | MVTec AD Screw Dataset"
)
