import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

MODEL_PATH = "receipt_orientation_model.pth"

# Inference transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_inference_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    return model, idx_to_class

def predict_single_image(model, idx_to_class, img_path):
    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()

    return idx_to_class[pred_idx], confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 predict_folder.py <path_to_images_folder>")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: {target_dir} is not a valid directory.")
        sys.exit(1)

    model, idx_to_class = load_inference_model()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in valid_exts])

    print(f"\nEvaluating {len(images)} images in '{target_dir}'...\n")
    print(f"{'Filename':<35} | {'Predicted Orientation':<22} | {'Confidence'}")
    print("-" * 75)

    for p in images:
        pred_angle, conf = predict_single_image(model, idx_to_class, p)
        print(f"{p.name:<35} | {pred_angle + '°':<22} | {conf * 100:.1f}%")

if __name__ == "__main__":
    main()