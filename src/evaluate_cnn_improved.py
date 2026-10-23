
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "neu_split" / "test"

MODEL_PATH = PROJECT_ROOT / "models" / "neu_cnn_improved.pth"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 200
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# 2. IMAGE TRANSFORMATION
# ==========================================

test_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor()

])


# ==========================================
# 3. LOAD TEST DATA
# ==========================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print("\nClasses:", class_names)

print(
    "Test images:",
    len(test_dataset)
)


# ==========================================
# 4. IMPROVED CNN ARCHITECTURE
# ==========================================

class ImprovedDefectCNN(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 2

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 3

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 4

            nn.Conv2d(
                64,
                128,
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
# 5. LOAD TRAINED MODEL
# ==========================================

print("\nLoading improved CNN model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Use the class names saved during training

saved_class_names = checkpoint["class_names"]

print(
    "Saved model classes:",
    saved_class_names
)

if saved_class_names != class_names:

    raise ValueError(
        "Class names do not match between model and test dataset."
    )

model = ImprovedDefectCNN(
    num_classes=len(saved_class_names)
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "\nImproved CNN model loaded successfully."
)


# ==========================================
# 6. PREDICT TEST IMAGES
# ==========================================

all_labels = []

all_predictions = []

print("\nEvaluating test dataset...\n")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ==========================================
# 7. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ==========================================
# 8. DISPLAY FINAL RESULTS
# ==========================================

print("\n")

print("=" * 55)

print("      IMPROVED CNN TEST RESULTS")

print("=" * 55)

print(
    f"\nAccuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-score:  {f1:.4f}"
)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

report = classification_report(

    all_labels,

    all_predictions,

    labels=list(range(len(class_names))),

    target_names=class_names,

    zero_division=0

)

print("\nClassification Report:\n")

print(report)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=list(range(len(class_names)))

)

print("\nConfusion Matrix:\n")

print(cm)


# ==========================================
# 11. SAVE METRICS
# ==========================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Weighted Precision",

        "Weighted Recall",

        "Weighted F1-score"

    ],

    "Value": [

        accuracy,

        precision,

        recall,

        f1

    ]

})

metrics_df.to_csv(

    RESULTS_DIR / "neu_cnn_improved_test_metrics.csv",

    index=False

)


# ==========================================
# 12. SAVE CLASSIFICATION REPORT
# ==========================================

report_dict = classification_report(

    all_labels,

    all_predictions,

    labels=list(range(len(class_names))),

    target_names=class_names,

    output_dict=True,

    zero_division=0

)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(

    RESULTS_DIR / "neu_cnn_improved_classification_report.csv"

)


# ==========================================
# 13. SAVE CONFUSION MATRIX
# ==========================================

plt.figure(
    figsize=(9, 7)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names

)

plt.title(
    "Improved CNN - Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "neu_cnn_improved_confusion_matrix.png",

    dpi=300

)

plt.close()


# ==========================================
# 14. SAVE TEST PREDICTIONS
# ==========================================

prediction_rows = []

for index, (true_label, predicted_label) in enumerate(

    zip(
        all_labels,
        all_predictions
    )

):

    image_path = test_dataset.samples[index][0]

    prediction_rows.append({

        "image": image_path,

        "true_class": class_names[true_label],

        "predicted_class": class_names[predicted_label],

        "correct": true_label == predicted_label

    })


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_df.to_csv(

    RESULTS_DIR / "neu_cnn_improved_test_predictions.csv",

    index=False

)


# ==========================================
# 15. FINAL SUMMARY
# ==========================================

print("\n")

print("=" * 55)

print("IMPROVED CNN EVALUATION COMPLETED")

print("=" * 55)

print("\nSaved files:")

print(
    "1. neu_cnn_improved_test_metrics.csv"
)

print(
    "2. neu_cnn_improved_classification_report.csv"
)

print(
    "3. neu_cnn_improved_confusion_matrix.png"
)

print(
    "4. neu_cnn_improved_test_predictions.csv"
)

print("\nAll results saved in the results folder.")
