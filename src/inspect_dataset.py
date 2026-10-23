
from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/mvtec/metal_nut")

for split in ["train", "test", "ground_truth"]:
    split_dir = DATA_DIR / split
    print(f"\n{split.upper()}")

    if not split_dir.exists():
        print("Folder not found:", split_dir)
        continue

    for category_dir in sorted(split_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        images = [
            p for p in category_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        ]

        print(f"{category_dir.name}: {len(images)} images")

        if images:
            try:
                with Image.open(images[0]) as img:
                    print(
                        "  Example:",
                        images[0].name,
                        "| Size:",
                        img.size,
                        "| Mode:",
                        img.mode
                    )
            except Exception as error:
                print("  Could not read example:", error)
