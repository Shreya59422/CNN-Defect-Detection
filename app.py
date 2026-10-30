
from pathlib import Path
import tempfile

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

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

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("CNN-Based Defect Detection in Manufacturing Processes")
st.write(
    "AI-assisted visual inspection prototype using "
    "PatchCore anomaly detection."
)

st.info(
    "This prototype is configured for the MVTec AD screw category. "
    "It uses a model trained on normal images."
)


# --------------------------------------------------
# LOAD MODEL
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
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a screw/component image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded component")
        st.image(image, use_container_width=True)

    if st.button("Inspect Component", type="primary"):

        try:
            model, engine = load_model()

            with st.spinner("Analysing component..."):

                # Save uploaded image temporarily
                with tempfile.TemporaryDirectory() as temp_dir:

                    image_path = Path(temp_dir) / "uploaded_image.png"
                    image.save(image_path)

                    # Anomalib handles preprocessing internally
                    dataset = PredictDataset(
                        path=temp_dir,
                        image_size=(256, 256),
                    )

                    predictions = engine.predict(
                        model=model,
                        dataset=dataset,
                        ckpt_path=str(CHECKPOINT),
                        return_predictions=True,
                    )

                if not predictions:
                    st.error("No prediction was returned.")
                    st.stop()

                batch = predictions[0]

                # Extract prediction values
                anomaly_score = float(
                    batch.pred_score[0].detach().cpu().item()
                )

                predicted_label = int(
                    batch.pred_label[0].detach().cpu().item()
                )

                anomaly_map = (
                    batch.anomaly_map[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )

                # Resize anomaly map for display
                anomaly_resized = np.asarray(
                    Image.fromarray(
                        anomaly_map.astype(np.float32)
                    ).resize(
                        image.size,
                        Image.Resampling.BILINEAR,
                    )
                )

                # Normalize only for visualization
                heatmap = (
                    (anomaly_resized - anomaly_resized.min())
                    / (
                        anomaly_resized.max()
                        - anomaly_resized.min()
                        + 1e-8
                    )
                )

                # Create heatmap overlay
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.imshow(image)
                ax.imshow(
                    heatmap,
                    cmap="jet",
                    alpha=0.45,
                )
                ax.axis("off")
                ax.set_title("Anomaly Heatmap Overlay")
                plt.tight_layout()

                # Display results
                st.divider()
                st.subheader("Inspection Results")

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric(
                        "Anomaly Score",
                        f"{anomaly_score:.4f}",
                    )

                with result_col2:
                    if predicted_label == 1:
                        st.error("Suspicious component detected")
                    else:
                        st.success("Component classified as normal")

                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "The heatmap highlights regions with higher "
                    "anomaly scores. It is a visual aid, not a "
                    "verified defect segmentation."
                )

                st.warning(
                    "Inspect suspicious predictions manually. "
                    "The model may miss defects, and a normal "
                    "prediction does not guarantee a defect-free part."
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.exception(error)

else:
    st.caption("Upload an image to begin inspection.")
