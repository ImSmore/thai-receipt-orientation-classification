import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 4
MODEL_SAVE_PATH = "receipt_orientation_model.pth"

# Device setup: Apple Silicon (MPS) -> CUDA -> CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Data Transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, class_names, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.float() / dataset_sizes[phase]).item()

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc * 100:.2f}%")

    model.load_state_dict(best_model_wts)
    return model, history

def main():
    print(f"Using device: {device}")

    # num_workers=0 avoids multiprocessing spawn crashes on macOS
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == "train"), num_workers=0)
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    print(f"Classes detected: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # ResNet-18 Transfer Learning
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, class_names, num_epochs=NUM_EPOCHS
    )

    # Save Checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": image_datasets["train"].class_to_idx,
        "classes": class_names
    }, MODEL_SAVE_PATH)
    print(f"Saved best model weights to {MODEL_SAVE_PATH}")

    # Plot & Save Report Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Val Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Saved training curves as 'training_results.png'")

if __name__ == "__main__":
    main()