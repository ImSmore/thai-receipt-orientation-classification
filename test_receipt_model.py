"""
Test a receipt-orientation model locally, whichever format it was saved in.

Handles two kinds of .pth files automatically:
  1. TorchScript trace (e.g. thai_receipt_orientation_EXAM.pth) -> loaded with torch.jit.load
  2. Raw state-dict checkpoint (e.g. receipt_orientation_model.pth, or
     thai_receipt_orientation_training_checkpoint.pth) -> architecture is rebuilt and
     the weights are loaded with load_state_dict. Use --arch to tell it which
     architecture the checkpoint was trained with.

Setup (once):
    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision pillow

Usage:
    # TorchScript model - architecture is auto-detected, no --arch needed
    python test_receipt_model.py --model thai_receipt_orientation_EXAM.pth --folder test_images/

    # Raw ResNet18 checkpoint
    python test_receipt_model.py --model receipt_orientation_model.pth --arch resnet18 --folder test_images/

    # Raw EfficientNetV2-M checkpoint
    python test_receipt_model.py --model thai_receipt_orientation_training_checkpoint.pth --arch efficientnet_v2_m --folder test_images/

    # Single image + save folder results to CSV
    python test_receipt_model.py --model thai_receipt_orientation_EXAM.pth --image receipt.jpg
    python test_receipt_model.py --model thai_receipt_orientation_EXAM.pth --folder test_images/ --csv results.csv
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torchvision  # noqa: F401  -- required so torch.jit.load can resolve torchvision ops
from torchvision import models, transforms
from PIL import Image, ImageOps

DEFAULT_CLASS_ANGLES = [0, 90, 180, 270]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LoadedModel:
    """Wraps either a TorchScript model or a rebuilt raw-checkpoint model behind one interface."""

    def __init__(self, model, idx_to_class, preprocess_fn):
        self.model = model
        self.idx_to_class = idx_to_class   # dict: class index -> label (e.g. angle in degrees)
        self.preprocess = preprocess_fn    # function: PIL.Image -> input tensor [1, 3, H, W]

    def predict(self, image_path):
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)  # correct phone-camera EXIF rotation metadata
            img = img.convert('RGB')
            tensor = self.preprocess(img)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        all_probs = {self.idx_to_class[i]: float(probs[i].item()) for i in range(len(probs))}
        return self.idx_to_class[pred_idx], confidence, all_probs


def torchscript_preprocess(img, image_size):
    # Resize with PIL first (same method + speed as training) -- the traced model now
    # only normalizes, it no longer resizes internally.
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return transforms.functional.to_tensor(img).unsqueeze(0)


def make_checkpoint_preprocess(image_size):
    pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return lambda img: pipeline(img).unsqueeze(0)


def load_model(model_path, arch):
    model_path = str(model_path)

    # Try TorchScript first -- if it fails, fall back to a raw checkpoint.
    try:
        ts_model = torch.jit.load(model_path, map_location='cpu')
        ts_model.eval()
        # The exported model expects pre-resized input; it stores the target size itself.
        image_size = int(ts_model.image_size.item()) if hasattr(ts_model, 'image_size') else 224
        idx_to_class = {i: angle for i, angle in enumerate(DEFAULT_CLASS_ANGLES)}
        preprocess_fn = lambda img, _size=image_size: torchscript_preprocess(img, _size)
        return LoadedModel(ts_model, idx_to_class, preprocess_fn)
    except RuntimeError:
        pass

    # Raw state-dict checkpoint.
    checkpoint = torch.load(model_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"'{model_path}' isn't a TorchScript file and doesn't look like a "
            "recognized checkpoint (no 'model_state_dict' key)."
        )

    if arch is None:
        raise ValueError(
            f"'{model_path}' is a raw checkpoint, not TorchScript. "
            "Pass --arch resnet18 or --arch efficientnet_v2_m so the architecture can be rebuilt."
        )

    # Figure out class labels from whatever the checkpoint provides.
    if 'class_to_idx' in checkpoint:
        idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        idx_to_class = {i: int(label) if str(label).isdigit() else label for i, label in idx_to_class.items()}
    elif 'class_angles' in checkpoint:
        idx_to_class = {i: angle for i, angle in enumerate(checkpoint['class_angles'])}
    else:
        idx_to_class = {i: angle for i, angle in enumerate(DEFAULT_CLASS_ANGLES)}

    num_classes = len(idx_to_class)

    if arch == 'resnet18':
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        preprocess_fn = make_checkpoint_preprocess(checkpoint.get('image_size', 224))
    elif arch == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m()
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        preprocess_fn = make_checkpoint_preprocess(checkpoint.get('image_size', 480))
    else:
        raise ValueError(f'Unsupported --arch: {arch}')

    model.eval()
    return LoadedModel(model, idx_to_class, preprocess_fn)


def find_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description='Test a receipt orientation model (TorchScript or raw checkpoint).')
    parser.add_argument('--model', required=True, help='Path to the .pth file')
    parser.add_argument('--arch', choices=['resnet18', 'efficientnet_v2_m'], default=None,
                         help='Architecture to rebuild -- only needed for raw (non-TorchScript) checkpoints')
    parser.add_argument('--image', help='Path to a single image to test')
    parser.add_argument('--folder', help='Path to a folder of images to test')
    parser.add_argument('--csv', help='Optional: save folder results to this CSV path')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error('Provide either --image or --folder')

    loaded = load_model(args.model, args.arch)

    if args.image:
        angle, confidence, all_probs = loaded.predict(args.image)
        print(f'{Path(args.image).name}')
        print(f'  Predicted rotation: {angle}°  (confidence: {confidence:.1%})')
        print('  All class probabilities:')
        for cls_angle, p in all_probs.items():
            print(f'    {cls_angle}°: {p:.1%}')

    if args.folder:
        images = find_images(args.folder)
        if not images:
            print(f'No images found in {args.folder}')
            return

        results = []
        print(f'Testing {len(images)} images from {args.folder}\n')
        print(f'{"File":<35} {"Predicted":>10} {"Confidence":>12}')
        print('-' * 60)

        for image_path in images:
            angle, confidence, all_probs = loaded.predict(image_path)
            results.append({
                'file': image_path.name,
                'predicted_angle': angle,
                'confidence': round(confidence, 4),
                **{f'prob_{a}': round(p, 4) for a, p in all_probs.items()},
            })
            flag = '  <-- low confidence' if confidence < 0.6 else ''
            print(f'{image_path.name:<35} {str(angle)+"°":>10} {confidence:>11.1%}{flag}')

        avg_conf = sum(r['confidence'] for r in results) / len(results)
        print('-' * 60)
        print(f'Average confidence: {avg_conf:.1%}')

        if args.csv:
            fieldnames = list(results[0].keys())
            with open(args.csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f'\nSaved results to {args.csv}')


if __name__ == '__main__':
    main()