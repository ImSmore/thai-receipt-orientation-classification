import sys
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

def predict_orientation(image_path):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    # Rebuild model architecture
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}

    # Load test image
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()

    predicted_angle = idx_to_class[pred_idx]
    return predicted_angle, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_receipt_image>")
        sys.exit(1)

    img_file = sys.argv[1]
    angle, conf = predict_orientation(img_file)
    print(f"Image: {img_file} | Predicted Orientation: {angle}° (Confidence: {conf*100:.1f}%)")