import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# ================= CONFIG =================
TRAIN_DIR = "D:/Final Year project/datasets/train"
TEST_DIR  = "D:/Final Year project/datasets/test"

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ================= TRANSFORMS =================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= DATASETS =================
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
test_ds  = datasets.ImageFolder(TEST_DIR, transform=test_tfms)

class_names = train_ds.classes
num_classes = len(class_names)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ================= COMMON FUNCTIONS =================
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(y_true, y_pred):
    return (np.array(y_true) == np.array(y_pred)).mean() * 100

def train_epoch(model, loader, optimizer, scaler):
    model.train()
    loss_sum = 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
    return loss_sum / len(loader)

def eval_epoch(model, loader):
    model.eval()
    loss_sum = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            loss_sum += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return loss_sum / len(loader), y_true, y_pred

# ================= TRAINING FUNCTION =================
def train_model(model_name):
    print(f"\n===== Training {model_name} =====")

    if model_name == "B3":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    else:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.features[-3:].parameters():
        p.requires_grad = True

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    model.to(DEVICE)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scaler = torch.amp.GradScaler("cuda")

    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        test_loss, y_true, y_pred = eval_epoch(model, test_loader)
        acc = calculate_accuracy(y_true, y_pred)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(acc)

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Acc: {acc:.2f}%")

    return model, train_losses, test_losses, test_accs, y_true, y_pred

# ================= TRAIN B3 =================
model_b3, tr_b3, te_b3, acc_b3, y_true, y_pred = train_model("B3")

# ================= TRAIN B0 (for comparison) =================
model_b0, tr_b0, te_b0, acc_b0, _, _ = train_model("B0")

# ================= CONFUSION MATRIX (B3) =================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix – EfficientNet-B3")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nClassification Report (B3):")
print(classification_report(y_true, y_pred, target_names=class_names))

# ================= 1️⃣ LOSS GAIN PLOT =================
loss_gain = [(te_b3[0] - l) / te_b3[0] * 100 for l in te_b3]

plt.figure(figsize=(8,5))
plt.plot(loss_gain)
plt.xlabel("Epochs")
plt.ylabel("Loss Gain (%)")
plt.title("Test Loss Gain – EfficientNet-B3")
plt.grid(True)
plt.show()

# ================= 2️⃣ SMOOTHED GAIN (Moving Avg) =================
window = 5
smooth_gain = np.convolve(loss_gain, np.ones(window)/window, mode='valid')

plt.figure(figsize=(8,5))
plt.plot(smooth_gain)
plt.xlabel("Epochs")
plt.ylabel("Smoothed Loss Gain (%)")
plt.title("Smoothed Loss Gain (Moving Average)")
plt.grid(True)
plt.show()

# ================= 3️⃣ ACCURACY GAIN =================
acc_gain_b3 = [a - acc_b3[0] for a in acc_b3]

plt.figure(figsize=(8,5))
plt.plot(acc_gain_b3)
plt.xlabel("Epochs")
plt.ylabel("Accuracy Gain (%)")
plt.title("Accuracy Gain – EfficientNet-B3")
plt.grid(True)
plt.show()

# ================= 4️⃣ COMPARISON GAIN (B0 vs B3) =================
acc_gain_b0 = [a - acc_b0[0] for a in acc_b0]

plt.figure(figsize=(8,5))
plt.plot(acc_gain_b0, label="EfficientNet-B0")
plt.plot(acc_gain_b3, label="EfficientNet-B3")
plt.xlabel("Epochs")
plt.ylabel("Accuracy Gain (%)")
plt.title("Accuracy Gain Comparison")
plt.legend()
plt.grid(True)
plt.show()

# ================= 5️⃣ CUMULATIVE GAIN =================
cum_gain_b3 = np.cumsum(acc_gain_b3)

plt.figure(figsize=(8,5))
plt.plot(cum_gain_b3)
plt.xlabel("Epochs")
plt.ylabel("Cumulative Accuracy Gain")
plt.title("Cumulative Gain – EfficientNet-B3")
plt.grid(True)
plt.show()

# ================= SAVE MODEL =================
torch.save(model_b3.state_dict(), "efficientnet_b3_final.pth")
print("\nModel saved: efficientnet_b3_final.pth")
torch.save(model_b0.state_dict(), "efficientnet_b0_final.pth")
print("\nModel saved: efficientnet_b0_final.pth")
