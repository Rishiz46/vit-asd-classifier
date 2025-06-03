# Imports and Configuration
import os, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from PIL import Image

warnings.filterwarnings("ignore")

CSV_PATH = "/kaggle/input/abide-dataset-csv/Phenotypic_V1_0b_preprocessed1.csv"
NPY_DIR = "/kaggle/input/abide-preprocessed-npy700-files"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 10
LEARNING_RATE = 1e-5
MODEL_SAVE_PATH = "/kaggle/working/vit_model"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CSV and match NPY files
df = pd.read_csv(CSV_PATH)
df = df[df['SITE_ID'].notna() & df['DX_GROUP'].isin([1, 2])]
npy_files = os.listdir(NPY_DIR)

matched = []
for _, row in df.iterrows():
    sid = str(row['SUB_ID']).zfill(7)
    prefix = f"{row['SITE_ID']}_{sid}".lower()
    file = next((f for f in npy_files if f.lower().startswith(prefix)), None)
    if file:
        matched.append({"label": int(row['DX_GROUP']) - 1, "npy_path": os.path.join(NPY_DIR, file)})

df_matched = pd.DataFrame(matched)
print("Matched files:", len(df_matched))

# Train-val split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(split.split(df_matched, df_matched['label']))
train_df, val_df = df_matched.iloc[train_idx], df_matched.iloc[val_idx]

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter()], p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Image processor
extractor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)

# Dataset class
class FMriDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['npy_path'])
        mid = data.shape[0] // 2
        slices = data[mid - 2: mid + 3]
        img_3ch = np.stack([slices.mean(0)] * 3, axis=-1)
        img_3ch = ((img_3ch - img_3ch.min()) / (img_3ch.ptp() + 1e-8) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_3ch, mode='RGB')
        img = self.transform(pil_img)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        inputs = extractor(images=img_np, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.df)

train_loader = DataLoader(FMriDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FMriDataset(val_df, val_transform), batch_size=BATCH_SIZE)
print("Dataloader initialized")

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        logp = self.ce(logits, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

loss_fn = FocalLoss()

# Model definition
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training loop
best_val_acc = 0
epochs_no_improve = 0
print("Training started")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs).logits
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_model.pth"))
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Control", "ASD"]))
