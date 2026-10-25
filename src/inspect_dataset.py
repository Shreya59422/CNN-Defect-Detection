
from pathlib import Path
from PIL import Image


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# NEU dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "neu_split"

# Dataset splits
SPLITS = ["train", "val", "test"]

# Six NEU defect classes
CLASSES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]

# Supported image formats
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# DATASET INSPECTION
# --------------------------------------------------

def inspect_dataset():

    total_images = 0

    print("=" * 60)
    print("NEU STEEL DEFECT DATASET INSPECTION")
    print("=" * 60)

    print("Project directory:", PROJECT_ROOT)
    print("Dataset directory:", DATA_DIR)

    if not DATA_DIR.exists():

        print("\nDataset directory not found!")

        return

    for split in SPLITS:

        split_dir = DATA_DIR / split

        print(f"\n{split.upper()}")
        print("-" * 40)

        if not split_dir.exists():

            print("Folder not found:", split_dir)

            continue

        split_total = 0

        for class_name in CLASSES:

            class_dir = split_dir / class_name

            if not class_dir.exists():

                print(f"{class_name}: Folder missing")

                continue

            images = [

                path

                for path in class_dir.rglob("*")

                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS

            ]

            image_count = len(images)

            split_total += image_count

            print(f"{class_name}: {image_count} images")

            # Display example image information
            if images:

                try:

                    with Image.open(images[0]) as img:

                        print(
                            "  Example:",
                            images[0].name,
                            "| Size:",
                            img.size,
                            "| Mode:",
                            img.mode,
                        )

                except Exception as error:

                    print(
                        "  Could not read example:",
                        error,
                    )

        print(f"\nTotal {split}: {split_total}")

        total_images += split_total

    print("\n" + "=" * 60)
    print("TOTAL DATASET IMAGES:", total_images)
    print("=" * 60)


# --------------------------------------------------
# CHECK CORRUPTED IMAGES
# --------------------------------------------------

def check_corrupted_images():

    print("\n" + "=" * 60)
    print("CHECKING IMAGE FILES")
    print("=" * 60)

    if not DATA_DIR.exists():

        print("Dataset directory not found:", DATA_DIR)

        return

    corrupted = []

    checked = 0

    for split in SPLITS:

        split_dir = DATA_DIR / split

        if not split_dir.exists():

            continue

        for class_dir in split_dir.iterdir():

            if not class_dir.is_dir():

                continue

            for path in class_dir.rglob("*"):

                if not path.is_file():

                    continue

                if path.suffix.lower() not in IMAGE_EXTENSIONS:

                    continue

                checked += 1

                try:

                    with Image.open(path) as img:

                        img.verify()

                except Exception as error:

                    corrupted.append(
                        (path, str(error))
                    )

    print("Images checked:", checked)

    print("Corrupted images:", len(corrupted))

    if corrupted:

        print("\nCorrupted files:")

        for path, error in corrupted:

            print(path, "|", error)

    else:

        print("All images are readable!")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    inspect_dataset()

    check_corrupted_images()
