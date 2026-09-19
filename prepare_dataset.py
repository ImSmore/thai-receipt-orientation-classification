import os
import random
from pathlib import Path
from PIL import Image, ImageOps

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent / "dataset"
SRC_DIR = BASE_DIR / "raw_originals"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Split ratio (80% train, 20% validation)
VAL_RATIO = 0.2
RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Pillow transpose constants for CW rotation:
# Image.ROTATE_270 rotates CCW by 270 deg == 90 deg CW
# Image.ROTATE_90 rotates CCW by 90 deg == 270 deg CW
ROTATIONS = {
    "0": None,
    "90": Image.Transpose.ROTATE_270,
    "180": Image.Transpose.ROTATE_180,
    "270": Image.Transpose.ROTATE_90,
}

def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source folder not found: {SRC_DIR}")

    # Gather valid image files
    image_files = [f for f in SRC_DIR.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
    total_images = len(image_files)
    
    print(f"Found {total_images} original images in {SRC_DIR}")
    if total_images < 200:
        print(f"WARNING: The project requires at least 200 original images. Currently found: {total_images}")

    # Shuffle and split original receipts first to avoid data leakage
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    split_idx = int(total_images * (1 - VAL_RATIO))
    splits = {
        "train": (TRAIN_DIR, image_files[:split_idx]),
        "val": (VAL_DIR, image_files[split_idx:])
    }

    print(f"Training originals: {len(splits['train'][1])} (x4 = {len(splits['train'][1]) * 4} images)")
    print(f"Validation originals: {len(splits['val'][1])} (x4 = {len(splits['val'][1]) * 4} images)")

    # Process images and generate rotations
    for split_name, (dest_root, files) in splits.items():
        # Create output class directories: 0, 90, 180, 270
        for class_label in ROTATIONS.keys():
            (dest_root / class_label).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            stem = img_path.stem
            try:
                with Image.open(img_path) as img:
                    # Fix EXIF orientation tag from smartphones
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")

                    # Generate each orientation class
                    for label, transpose_op in ROTATIONS.items():
                        out_path = dest_root / label / f"{stem}_rot{label}.jpg"
                        
                        if transpose_op is not None:
                            rotated = img.transpose(transpose_op)
                        else:
                            rotated = img

                        rotated.save(out_path, format="JPEG", quality=95)

            except Exception as e:
                print(f"Skipping corrupt file {img_path.name}: {e}")

    print("\nDataset preparation complete! Ready for torchvision.datasets.ImageFolder.")

if __name__ == "__main__":
    main()