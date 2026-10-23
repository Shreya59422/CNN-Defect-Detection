
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "neu_cnn_test_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "neu_cnn_misclassified_gallery.png"
)


# ==========================================
# 2. LOAD PREDICTIONS
# ==========================================

df = pd.read_csv(PREDICTIONS_FILE)

errors = df[
    df["correct"] == False
].copy()

print("\nTotal test images:", len(df))
print("Misclassified images:", len(errors))


# ==========================================
# 3. CREATE GALLERY
# ==========================================

if errors.empty:

    print("\nNo misclassified images found.")

else:

    columns = 3

    rows = (
        len(errors) + columns - 1
    ) // columns

    # Fixed-size figure

    fig = plt.figure(
        figsize=(15, rows * 4.5)
    )

    # Main title

    fig.suptitle(
        "Misclassified Surface Defect Images",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    # ==========================================
    # 4. CREATE SEPARATE IMAGE AND TEXT PANELS
    # ==========================================

    for index, (_, row) in enumerate(
        errors.iterrows()
    ):

        row_number = index // columns

        column_number = index % columns

        # Position of each gallery item

        left = (
            0.04
            + column_number * 0.32
        )

        bottom = (
            0.08
            + (rows - 1 - row_number)
            * 0.29
        )

        width = 0.28

        image_height = 0.19

        text_height = 0.075

        # --------------------------------------
        # IMAGE PANEL
        # --------------------------------------

        image_ax = fig.add_axes(
            [
                left,
                bottom + text_height,
                width,
                image_height
            ]
        )

        image_path = Path(
            row["image"]
        )

        image = Image.open(
            image_path
        ).convert("L")

        image_ax.imshow(
            image,
            cmap="gray"
        )

        image_ax.axis("off")

        # --------------------------------------
        # TEXT PANEL
        # --------------------------------------

        text_ax = fig.add_axes(
            [
                left,
                bottom,
                width,
                text_height
            ]
        )

        text_ax.axis("off")

        text_ax.text(
            0.5,
            0.85,
            f"Actual: {row['true_class']}",
            ha="center",
            va="center",
            fontsize=11,
            transform=text_ax.transAxes
        )

        text_ax.text(
            0.5,
            0.50,
            f"Predicted: {row['predicted_class']}",
            ha="center",
            va="center",
            fontsize=11,
            transform=text_ax.transAxes
        )

        text_ax.text(
            0.5,
            0.15,
            image_path.name,
            ha="center",
            va="center",
            fontsize=9,
            transform=text_ax.transAxes
        )

    # ==========================================
    # 5. SAVE GALLERY
    # ==========================================

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nGallery generated successfully!")

    print("Saved at:")
    print(OUTPUT_FILE)

    plt.show()


# ==========================================
# 6. PRINT ERROR SUMMARY
# ==========================================

print("\nMisclassification Summary:")

if not errors.empty:

    summary = (
        errors.groupby(
            [
                "true_class",
                "predicted_class"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

print("\nError analysis completed.")
