import os
import shutil
import zipfile
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent / "dataset"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Output submission folder & zip name
SUBMISSION_DIR = BASE_DIR / "submission_dataset" / "dataset"
ZIP_OUTPUT_PATH = Path(__file__).resolve().parent / "receipt_dataset.zip"

CLASSES = ["0", "90", "180", "270"]

def main():
    print("=== Packaging Dataset for Submission ===")
    
    # 1. Clean previous run if it exists
    if SUBMISSION_DIR.parent.exists():
        shutil.rmtree(SUBMISSION_DIR.parent)
    
    for cls in CLASSES:
        (SUBMISSION_DIR / cls).mkdir(parents=True, exist_ok=True)

    # 2. Copy images from both train and val into the unified structure
    total_copied = 0
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist!")
            continue

        for cls in CLASSES:
            src_cls_dir = split_dir / cls
            if not src_cls_dir.exists():
                continue

            for img_file in src_cls_dir.iterdir():
                if img_file.is_file() and not img_file.name.startswith("."):
                    dest_file = SUBMISSION_DIR / cls / img_file.name
                    shutil.copy2(img_file, dest_file)
                    total_copied += 1

    print(f"Collected total of {total_copied} images across all classes:")
    for cls in CLASSES:
        count = len(list((SUBMISSION_DIR / cls).iterdir()))
        print(f"  Class {cls}°: {count} images")

    # 3. Create the ZIP archive
    print(f"\nCompressing into '{ZIP_OUTPUT_PATH.name}'...")
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the submission folder so the top folder inside the zip is 'dataset/'
        root_dir = SUBMISSION_DIR.parent  # contains 'dataset/'
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.startswith("."):
                    continue
                file_path = Path(root) / file
                archive_name = file_path.relative_to(root_dir)
                zipf.write(file_path, archive_name)

    # Clean up temporary unzipped folder
    shutil.rmtree(SUBMISSION_DIR.parent)

    zip_size_mb = ZIP_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone! Created: {ZIP_OUTPUT_PATH} ({zip_size_mb:.2f} MB)")
    print("\nInside the ZIP:")
    print("dataset/")
    print("├── 0/")
    print("├── 90/")
    print("├── 180/")
    print("└── 270/")

if __name__ == "__main__":
    main()