
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

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "anomaly_detection"
    / "Patchcore"
    / "MVTecAD"
)

METRICS_FILE = PROJECT_ROOT / "results" / "patchcore_all_categories.csv"

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

CATEGORY_LABELS = {
    "bottle": "Bottle",
    "cable": "Cable",
    "capsule": "Capsule",
    "carpet": "Carpet",
    "grid": "Grid",
    "hazelnut": "Hazelnut",
    "leather": "Leather",
    "metal_nut": "Metal Nut",
    "pill": "Pill",
    "screw": "Screw",
    "tile": "Tile",
    "toothbrush": "Toothbrush",
    "transistor": "Transistor",
    "wood": "Wood",
    "zipper": "Zipper"
}

# Detailed screw synthetic test metrics.
# These must only be displayed for the screw experiment.

SCREW_METRICS = {
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

SCREW_CONFUSION_MATRIX = np.array([
    [38, 3],
    [24, 95]
])


# --------------------------------------------------
# 2. DISCOVER CATEGORY CHECKPOINTS
# --------------------------------------------------

def find_checkpoint(category):

    category_dir = CHECKPOINT_ROOT / category

    if not category_dir.exists():
        return None

    checkpoints = list(
        category_dir.glob("*/weights/lightning/model.ckpt")
    )

    if not checkpoints:
        return None

    # Prefer the highest version available.
    def version_number(path):
        version = path.parent.parent.parent.name

        try:
            return int(version.replace("v", ""))
        except ValueError:
            return -1

    return max(checkpoints, key=version_number)


CHECKPOINTS = {
    category: find_checkpoint(category)
    for category in CATEGORIES
}

AVAILABLE_CATEGORIES = [
    category
    for category, checkpoint in CHECKPOINTS.items()
    if checkpoint is not None
]


# --------------------------------------------------
# 3. LOAD CATEGORY-WISE METRICS
# --------------------------------------------------

@st.cache_data
def load_category_metrics():

    if not METRICS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(METRICS_FILE)

    if "Category" not in df.columns:
        return pd.DataFrame()

    df["Category"] = df["Category"].astype(str).str.lower()

    return df


CATEGORY_METRICS = load_category_metrics()


# --------------------------------------------------
# 4. CUSTOM DARK THEME
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
# 5. MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model(checkpoint_path):

    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    model = Patchcore()

    engine = Engine(
        default_root_dir=PROJECT_ROOT / "results" / "prototype"
    )

    return model, engine


# --------------------------------------------------
# 6. HELPER FUNCTIONS
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


def get_category_metrics(category):

    if CATEGORY_METRICS.empty:
        return None

    rows = CATEGORY_METRICS[
        CATEGORY_METRICS["Category"] == category
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def create_heatmap(original_image, anomaly_map):

    original = np.array(original_image.convert("RGB"))

    anomaly_map = np.asarray(
        anomaly_map,
        dtype=np.float32
    )

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


def inspect_image(uploaded_file, category):

    checkpoint = CHECKPOINTS.get(category)

    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint is available for {category}."
        )

    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.TemporaryDirectory() as temp_dir:

        image_path = Path(temp_dir) / "inspection_image.png"

        image.save(image_path)

        dataset = PredictDataset(
            path=temp_dir,
            image_size=(256, 256)
        )

        model, engine = load_model(str(checkpoint))

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=str(checkpoint),
            return_predictions=True
        )

    if not predictions:
        raise RuntimeError(
            "The model returned no predictions."
        )

    batch = predictions[0]

    score = float(batch.pred_score[0].item())
    label = int(batch.pred_label[0].item())

    anomaly_map = (
        batch.anomaly_map[0]
        .detach()
        .cpu()
        .numpy()
    )

    overlay = create_heatmap(image, anomaly_map)

    return image, score, label, anomaly_map, overlay


# --------------------------------------------------
# 7. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🔍 Inspection AI")

st.sidebar.caption(
    "CNN-Based Manufacturing Defect Detection"
)

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
st.sidebar.write("**Device:** CPU")

st.sidebar.write(
    f"**Available checkpoints:** {len(AVAILABLE_CATEGORIES)}/15"
)

selected_category = st.sidebar.selectbox(
    "Select Product Category",
    options=CATEGORIES,
    format_func=lambda x: CATEGORY_LABELS[x]
)

selected_checkpoint = CHECKPOINTS.get(selected_category)

if selected_checkpoint is not None:
    st.sidebar.success(
        f"{CATEGORY_LABELS[selected_category]} checkpoint available"
    )
    st.sidebar.caption(
        f"Checkpoint version: {selected_checkpoint.parent.parent.parent.name}"
    )
else:
    st.sidebar.error(
        f"No checkpoint found for {CATEGORY_LABELS[selected_category]}"
    )

st.sidebar.divider()

st.sidebar.caption(
    "Select a category to inspect its corresponding product."
)


# --------------------------------------------------
# 8. HEADER
# --------------------------------------------------

st.title("Industrial Defect Detection System")

st.caption(
    "AI-powered visual inspection across 15 MVTec AD categories"
)

st.divider()


# --------------------------------------------------
# 9. OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.subheader("Dashboard Overview")

    st.write(
        "Monitor model performance and inspect industrial "
        "components across multiple product categories."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Product Categories",
            str(len(AVAILABLE_CATEGORIES))
        )

    with col2:
        metric_card(
            "Available Checkpoints",
            f"{len(AVAILABLE_CATEGORIES)}/15"
        )

    with col3:
        if not CATEGORY_METRICS.empty:
            mean_auroc = CATEGORY_METRICS[
                "image_AUROC"
            ].mean()

            metric_card(
                "Mean Image AUROC",
                f"{mean_auroc * 100:.2f}%"
            )
        else:
            metric_card("Mean Image AUROC", "N/A")

    with col4:
        metric_card(
            "Selected Category",
            CATEGORY_LABELS[selected_category]
        )

    st.write("")

    st.markdown("### Category-wise Evaluation")

    if not CATEGORY_METRICS.empty:

        display_df = CATEGORY_METRICS.copy()

        display_df["Category"] = (
            display_df["Category"]
            .map(CATEGORY_LABELS)
            .fillna(display_df["Category"])
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            "Category metrics file was not found. "
            "Check results/patchcore_all_categories.csv."
        )

    st.info(
        "PatchCore learns from normal training images and "
        "identifies deviations during inspection. "
        "Suspicious predictions should be reviewed by a human inspector."
    )


# --------------------------------------------------
# 10. LIVE INSPECTION PAGE
# --------------------------------------------------

elif page == "Live Inspection":

    st.subheader("Live Component Inspection")

    st.write(
        f"Selected category: **{CATEGORY_LABELS[selected_category]}**"
    )

    if selected_checkpoint is None:

        st.error(
            "The checkpoint for this category is unavailable. "
            "Choose another category or add the required checkpoint."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload component image",
            type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_file is not None:

            original = Image.open(uploaded_file).convert("RGB")

            st.markdown("### Uploaded Image")
            st.image(original, use_container_width=True)

            if st.button(
                "Run Defect Inspection",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        f"Analysing {CATEGORY_LABELS[selected_category]}..."
                    ):

                        uploaded_file.seek(0)

                        image, score, label, anomaly_map, overlay = (
                            inspect_image(
                                uploaded_file,
                                selected_category
                            )
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
                            "Category": CATEGORY_LABELS[selected_category],
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
                        st.metric(
                            "Anomaly Score",
                            f"{score:.4f}"
                        )

                    with col2:
                        if label == 1:
                            st.error("Suspicious Component")
                        else:
                            st.success("Normal Component")

                    st.markdown("### Defect Localization")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Original Image")
                        st.image(
                            image,
                            use_container_width=True
                        )

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

                    st.error(
                        f"Inspection failed: {error}"
                    )


# --------------------------------------------------
# 11. MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.subheader("Multi-Category Model Evaluation")

    st.caption(
        "The following table contains category-wise evaluation "
        "results from patchcore_all_categories.csv."
    )

    if not CATEGORY_METRICS.empty:

        display_df = CATEGORY_METRICS.copy()

        display_df["Category"] = (
            display_df["Category"]
            .map(CATEGORY_LABELS)
            .fillna(display_df["Category"])
        )

        st.markdown("### Image-Level and Pixel-Level Metrics")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Image AUROC by Category")

        chart_df = CATEGORY_METRICS[
            ["Category", "image_AUROC"]
        ].copy()

        chart_df["Category"] = (
            chart_df["Category"]
            .map(CATEGORY_LABELS)
            .fillna(chart_df["Category"])
        )

        chart_df = chart_df.set_index("Category")

        st.bar_chart(chart_df)

        st.markdown("### Pixel F1-score by Category")

        pixel_df = CATEGORY_METRICS[
            ["Category", "pixel_F1Score"]
        ].copy()

        pixel_df["Category"] = (
            pixel_df["Category"]
            .map(CATEGORY_LABELS)
            .fillna(pixel_df["Category"])
        )

        pixel_df = pixel_df.set_index("Category")

        st.bar_chart(pixel_df)

    else:

        st.warning(
            "Category evaluation CSV is unavailable."
        )

    st.divider()

    st.markdown("### Detailed Screw Test Evaluation")

    st.caption(
        "These detailed metrics and the confusion matrix refer "
        "only to the separate screw synthetic test experiment."
    )

    columns = st.columns(3)

    screw_image_metrics = [
        ("Image AUROC", "95.51%"),
        ("Image Average Precision", "98.41%"),
        ("Image F1 Score", "87.56%"),
        ("Image Precision", "96.94%"),
        ("Image Recall", "79.83%"),
        ("False Negative Rate", "20.17%")
    ]

    for index, (name, value) in enumerate(screw_image_metrics):

        with columns[index % 3]:
            metric_card(name, value)

    st.markdown("### Screw Pixel-Level Metrics")

    columns = st.columns(3)

    screw_pixel_metrics = [
        ("Pixel AUROC", "98.89%"),
        ("Pixel Average Precision", "32.99%"),
        ("Pixel F1 Score", "2.28%"),
        ("Pixel Precision", "72.49%"),
        ("Pixel Recall", "1.16%")
    ]

    for index, (name, value) in enumerate(screw_pixel_metrics):

        with columns[index % 3]:
            metric_card(name, value)

    st.markdown("### Screw Image-Level Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 4))

    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    ax.imshow(
        SCREW_CONFUSION_MATRIX,
        cmap="Blues"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Normal", "Defective"],
        color="white"
    )

    ax.set_yticklabels(
        ["Normal", "Defective"],
        color="white"
    )

    ax.set_xlabel("Predicted Label", color="white")
    ax.set_ylabel("Actual Label", color="white")
    ax.set_title("Screw Confusion Matrix", color="white")

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(SCREW_CONFUSION_MATRIX[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold"
            )

    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "Screw pixel-level F1 and recall are low at the "
        "evaluated threshold. High pixel AUROC alone does not "
        "establish accurate defect localization."
    )


# --------------------------------------------------
# 12. INSPECTION HISTORY PAGE
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

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

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
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Multi-Category Industrial Defect Detection and Localization "
    "| PatchCore | MVTec AD | 15 Categories"
)
