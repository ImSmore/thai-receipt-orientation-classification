"""
Compare two receipt-orientation models (TorchScript or raw checkpoint, any mix)
across all four rotation classes, with per-class accuracy and confusion matrices.

Expects a data directory laid out like:
    dataset/val/0/    <- upright receipts
    dataset/val/90/
    dataset/val/180/
    dataset/val/270/

Usage:
    python compare_models.py \
        --model-a thai_receipt_orientation_EXAM.pth \
        --label-a EfficientNetV2-M \
        --model-b thai_receipt_orientation_resnet18.pth --arch-b resnet18 \
        --label-b ResNet18 \
        --data-dir dataset/val

    # Also save every individual prediction to a CSV:
    ... --csv comparison_results.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision  # noqa: F401  -- required so torch.jit.load can resolve torchvision ops
from torchvision import models, transforms
from PIL import Image, ImageOps

DEFAULT_CLASS_ANGLES = [0, 90, 180, 270]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model loading (handles both TorchScript exports and raw state-dict checkpoints)
# ---------------------------------------------------------------------------

class LoadedModel:
    def __init__(self, model, idx_to_class, preprocess_fn):
        self.model = model
        self.idx_to_class = idx_to_class
        self.preprocess = preprocess_fn

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
        return self.idx_to_class[pred_idx], confidence


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

    checkpoint = torch.load(model_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"'{model_path}' isn't a TorchScript file and doesn't look like a "
            "recognized checkpoint (no 'model_state_dict' key)."
        )

    # Fall back to the architecture recorded in the checkpoint itself, if present.
    if arch is None:
        arch = checkpoint.get('model_architecture')
    if arch is None:
        raise ValueError(
            f"'{model_path}' is a raw checkpoint, not TorchScript, and doesn't record its "
            "own architecture. Pass --arch-a/--arch-b resnet18 or efficientnet_v2_m."
        )

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
        raise ValueError(f'Unsupported architecture: {arch}')

    model.eval()
    return LoadedModel(model, idx_to_class, preprocess_fn)


def find_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


# ---------------------------------------------------------------------------
# Evaluation across all 4 class folders
# ---------------------------------------------------------------------------

def evaluate_model(loaded, data_dir, class_angles=DEFAULT_CLASS_ANGLES):
    class_to_idx = {angle: i for i, angle in enumerate(class_angles)}
    n = len(class_angles)
    confusion = [[0] * n for _ in range(n)]  # rows = true class, cols = predicted class
    records = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    total_correct = 0
    total = 0
    conf_sum = 0.0

    for angle in class_angles:
        folder = Path(data_dir) / str(angle)
        if not folder.exists():
            print(f'  Warning: folder not found, skipping: {folder}')
            continue

        for img_path in find_images(folder):
            pred_angle, confidence = loaded.predict(img_path)
            true_idx = class_to_idx[angle]
            pred_idx = class_to_idx.get(pred_angle)
            if pred_idx is not None:
                confusion[true_idx][pred_idx] += 1

            correct = (pred_angle == angle)
            class_total[angle] += 1
            total += 1
            conf_sum += confidence
            if correct:
                class_correct[angle] += 1
                total_correct += 1

            records.append({
                'true_angle': angle,
                'predicted_angle': pred_angle,
                'confidence': round(confidence, 4),
                'correct': correct,
                'file': str(img_path),
            })

    return {
        'confusion': confusion,
        'class_angles': class_angles,
        'class_correct': class_correct,
        'class_total': class_total,
        'total_correct': total_correct,
        'total': total,
        'avg_confidence': (conf_sum / total) if total else 0.0,
        'records': records,
    }


def print_summary(title, result):
    print(f'\n--- {title} ---')
    for a in result['class_angles']:
        c = result['class_correct'].get(a, 0)
        t = result['class_total'].get(a, 0)
        if t:
            print(f'  {a:>3}°: {c}/{t} correct ({c / t * 100:.1f}%)')
        else:
            print(f'  {a:>3}°: no images found')
    if result['total']:
        overall = result['total_correct'] / result['total'] * 100
        print(f'  Overall: {result["total_correct"]}/{result["total"]} correct ({overall:.1f}%)')
        print(f'  Average confidence: {result["avg_confidence"]:.1%}')


def print_confusion_matrix(title, result):
    angles = result['class_angles']
    print(f'\n{title} — confusion matrix (rows = true label, cols = predicted)')
    print('           ' + ''.join(f'{str(a) + "°":>8}' for a in angles))
    for i, a in enumerate(angles):
        row = ''.join(f'{result["confusion"][i][j]:>8}' for j in range(len(angles)))
        print(f'  true {str(a) + "°":>4} {row}')


def print_comparison_table(label_a, result_a, label_b, result_b):
    print('\n' + '=' * 66)
    print(f'{"Class":<10}{label_a:>28}{label_b:>28}')
    print('-' * 66)
    for a in result_a['class_angles']:
        ca, ta = result_a['class_correct'].get(a, 0), result_a['class_total'].get(a, 0)
        cb, tb = result_b['class_correct'].get(a, 0), result_b['class_total'].get(a, 0)
        acc_a = f'{ca}/{ta} ({ca / ta * 100:.1f}%)' if ta else 'n/a'
        acc_b = f'{cb}/{tb} ({cb / tb * 100:.1f}%)' if tb else 'n/a'
        print(f'{str(a) + "°":<10}{acc_a:>28}{acc_b:>28}')
    print('-' * 66)
    if result_a['total'] and result_b['total']:
        overall_a = f'{result_a["total_correct"]}/{result_a["total"]} ({result_a["total_correct"] / result_a["total"] * 100:.1f}%)'
        overall_b = f'{result_b["total_correct"]}/{result_b["total"]} ({result_b["total_correct"] / result_b["total"] * 100:.1f}%)'
        print(f'{"Overall":<10}{overall_a:>28}{overall_b:>28}')
    print('=' * 66)


def main():
    parser = argparse.ArgumentParser(description='Compare two receipt orientation models across all 4 classes.')
    parser.add_argument('--data-dir', default='dataset/val',
                         help="Folder containing subfolders '0', '90', '180', '270' (default: dataset/val)")
    parser.add_argument('--model-a', required=True)
    parser.add_argument('--arch-a', choices=['resnet18', 'efficientnet_v2_m'], default=None)
    parser.add_argument('--label-a', default=None, help='Display name for model A')
    parser.add_argument('--model-b', required=True)
    parser.add_argument('--arch-b', choices=['resnet18', 'efficientnet_v2_m'], default=None)
    parser.add_argument('--label-b', default=None, help='Display name for model B')
    parser.add_argument('--csv', help='Optional path to save every individual prediction from both models')
    args = parser.parse_args()

    label_a = args.label_a or Path(args.model_a).stem
    label_b = args.label_b or Path(args.model_b).stem

    print(f'Loading {label_a}: {args.model_a}')
    loaded_a = load_model(args.model_a, args.arch_a)
    print(f'Loading {label_b}: {args.model_b}')
    loaded_b = load_model(args.model_b, args.arch_b)

    print(f'\nEvaluating {label_a} on {args.data_dir} ...')
    result_a = evaluate_model(loaded_a, args.data_dir)
    print(f'Evaluating {label_b} on {args.data_dir} ...')
    result_b = evaluate_model(loaded_b, args.data_dir)

    print_summary(label_a, result_a)
    print_confusion_matrix(label_a, result_a)

    print_summary(label_b, result_b)
    print_confusion_matrix(label_b, result_b)

    print_comparison_table(label_a, result_a, label_b, result_b)

    if args.csv:
        fieldnames = ['model', 'true_angle', 'predicted_angle', 'confidence', 'correct', 'file']
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in result_a['records']:
                writer.writerow({'model': label_a, **rec})
            for rec in result_b['records']:
                writer.writerow({'model': label_b, **rec})
        print(f'\nSaved all individual predictions to {args.csv}')


if __name__ == '__main__':
    main()