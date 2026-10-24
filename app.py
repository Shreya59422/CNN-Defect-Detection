
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon="🔍",
    layout="centered"
)

# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/neu_cnn_improved.pth")

CLASS_NAMES = ["Cr", "In", "PS", "Pa", "RS", "Sc"]

DEFECT_NAMES = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "PS": "Pitted Surface",
    "Pa": "Patches",
    "RS": "Rolled-in Scale",
    "Sc": "Scratches"
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# CNN ARCHITECTURE
# ==========================================

class ImprovedDefectCNN(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 12 * 12,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = ImprovedDefectCNN(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model, class_names

# ==========================================
# IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (200, 200)
    ),

    transforms.ToTensor()
])

# ==========================================
# APPLICATION INTERFACE
# ==========================================

st.title("Manufacturing Defect Detection System")

st.subheader("Automated Surface Inspection")

st.write(
    "Upload a manufacturing surface image to identify and "
    "classify defects. The system provides the predicted "
    "defect category along with its confidence score "
    "to support quality inspection."
)
st.divider()

st.subheader("Upload Manufacturing Image")

uploaded_file = st.file_uploader(
    "Choose a surface image",
    type=["jpg", "jpeg", "png", "bmp"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Manufacturing Image",
        use_container_width=True
    )

    if st.button(
        "Predict Defect",
        type="primary"
    ):

        try:

            with st.spinner(
                "Analyzing image..."
            ):

                model, class_names = load_model()

                image_tensor = transform(
                    image
                ).unsqueeze(0)

                image_tensor = image_tensor.to(
                    DEVICE
                )

                with torch.no_grad():

                    outputs = model(
                        image_tensor
                    )

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    confidence, predicted_index = (
                        torch.max(
                            probabilities,
                            dim=1
                        )
                    )

                predicted_class = class_names[
                    predicted_index.item()
                ]

                confidence_value = (
                    confidence.item() * 100
                )

                defect_name = DEFECT_NAMES.get(
                    predicted_class,
                    predicted_class
                )

            # ----------------------------------
            # DISPLAY RESULT
            # ----------------------------------

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Defect: {defect_name}"
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence_value:.2f}%"
            )

            st.caption(
                "Confidence is the model's predicted "
                "class probability, not a guarantee "
                "of defect correctness."
            )

            # ----------------------------------
            # CLASS PROBABILITIES
            # ----------------------------------

            st.subheader("Class Probability Scores")

            probability_values = (
                probabilities[0]
                .cpu()
                .numpy()
            )

            for index, class_name in enumerate(
                class_names
            ):

                defect_name = DEFECT_NAMES.get(
                    class_name,
                    class_name
                )

                score = float(
                    probability_values[index]
                )

                st.write(
                    f"{defect_name}: "
                    f"{score * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

        except FileNotFoundError:

            st.error(
                "Model file not found. "
                "Check models/neu_cnn_improved.pth"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

# ==========================================
# FOOTER
# ==========================================

st.divider()
st.caption("Thank you for visiting!")
