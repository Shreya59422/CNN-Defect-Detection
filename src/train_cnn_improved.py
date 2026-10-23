
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "neu_split"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 200
BATCH_SIZE = 16
EPOCHS = 25

LEARNING_RATE = 0.0003
PATIENCE = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==========================================
# 2. DATA AUGMENTATION
# ==========================================

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(degrees=10),

    transforms.ToTensor(),

])


val_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

])


# ==========================================
# 3. LOAD DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)

print("\nClasses:", train_dataset.classes)

print("Training images:", len(train_dataset))

print("Validation images:", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 4. IMPROVED CNN MODEL
# ==========================================

class ImprovedDefectCNN(nn.Module):

    def __init__(self, num_classes=6):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 3
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 4
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2)

        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 12 * 12,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                256,
                num_classes
            )

        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================
# 5. INITIALIZE MODEL
# ==========================================

model = ImprovedDefectCNN(
    num_classes=len(train_dataset.classes)
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ==========================================
# 6. LEARNING RATE SCHEDULER
# ==========================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ==========================================
# 7. TRAINING FUNCTION
# ==========================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ==========================================
# 8. VALIDATION FUNCTION
# ==========================================

def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ==========================================
# 9. TRAINING LOOP
# ==========================================

best_val_accuracy = 0.0

best_val_loss = float("inf")

epochs_without_improvement = 0

history = []

MODEL_PATH = MODEL_DIR / "neu_cnn_improved.pth"

print("\nStarting improved CNN training...\n")


for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "val_loss": val_loss,

        "val_accuracy": val_accuracy,

        "learning_rate": current_lr

    })

    print(

        f"Epoch [{epoch + 1}/{EPOCHS}] "

        f"Train Loss: {train_loss:.4f} "

        f"Train Accuracy: {train_accuracy:.4f} "

        f"Val Loss: {val_loss:.4f} "

        f"Val Accuracy: {val_accuracy:.4f} "

        f"LR: {current_lr:.6f}"

    )


    # Save best validation accuracy model

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({

            "model_state_dict": model.state_dict(),

            "class_names": train_dataset.classes,

            "image_size": IMAGE_SIZE,

            "best_val_accuracy": best_val_accuracy,

            "model_type": "ImprovedDefectCNN"

        }, MODEL_PATH)

        print("Improved model saved.")

    
    # Early stopping based on validation loss

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ==========================================
# 10. SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(history)

history_df.to_csv(

    RESULTS_DIR / "neu_cnn_improved_training_history.csv",

    index=False

)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n===================================")

print("IMPROVED CNN TRAINING COMPLETED")

print("===================================")

print(

    "Best validation accuracy:",

    round(best_val_accuracy * 100, 2),

    "%"

)

print("Model saved:", MODEL_PATH)

print(

    "Training history saved:",

    RESULTS_DIR / "neu_cnn_improved_training_history.csv"

)
