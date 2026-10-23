
from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "neu"
OUTPUT_DIR = PROJECT_ROOT / "data" / "neu_split"

# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

# Check dataset folder
if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {SOURCE_DIR}"
    )

# Find BMP images
images = list(SOURCE_DIR.glob("*.bmp"))

if len(images) != 1800:
    raise ValueError(
        f"Expected 1800 images, found {len(images)}"
    )

# Create output directories
for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

# Split each class separately
for class_name in CLASSES:

    class_images = sorted(
        [
            image
            for image in images
            if image.stem.startswith(class_name + "_")
        ]
    )

    if len(class_images) != 300:
        raise ValueError(
            f"{class_name}: expected 300 images, "
            f"found {len(class_images)}"
        )

    # 70% train, 30% temporary
    train_images, temporary_images = train_test_split(
        class_images,
        test_size=1 - TRAIN_RATIO,
        random_state=42,
        shuffle=True
    )

    # Split remaining 30% equally into validation and test
    val_images, test_images = train_test_split(
        temporary_images,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    # Copy images into class folders
    for split_name, split_images in split_data.items():

        destination = OUTPUT_DIR / split_name / class_name

        for image_path in split_images:
            shutil.copy2(
                image_path,
                destination / image_path.name
            )

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Val={len(val_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset splitting completed.")

print(f"Dataset location: {OUTPUT_DIR}")
