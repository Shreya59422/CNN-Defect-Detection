
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

HISTORY_FILE = RESULTS_DIR / "neu_cnn_improved_training_history.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD TRAINING HISTORY
# ==========================================

history = pd.read_csv(HISTORY_FILE)

epochs = history["epoch"]


# ==========================================
# 3. TRAINING AND VALIDATION ACCURACY
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="s",
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.xticks(epochs)

plt.ylim(0, 1.05)

plt.grid(True)

plt.legend()

plt.tight_layout()

accuracy_path = RESULTS_DIR / "neu_cnn_accuracy_curve.png"

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Accuracy graph saved:", accuracy_path)


# ==========================================
# 4. TRAINING AND VALIDATION LOSS
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="s",
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.xticks(epochs)

plt.grid(True)

plt.legend()

plt.tight_layout()

loss_path = RESULTS_DIR / "neu_cnn_loss_curve.png"

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Loss graph saved:", loss_path)


# ==========================================
# 5. COMPLETION
# ==========================================

print("\nTraining graphs generated successfully.")
