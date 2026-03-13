import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from src.model import BaselineCNN
from src.dataset import GTSRBDataset, get_val_transforms
from torch.utils.data import DataLoader

CLASS_NAMES = [
    "Speed 20", "Speed 30", "Speed 50", "Speed 60", "Speed 70", "Speed 80",
    "End Speed 80", "Speed 100", "Speed 120", "No passing", "No pass >3.5t",
    "Right-of-way", "Priority road", "Yield", "Stop", "No vehicles",
    "Veh >3.5t prohib", "No entry", "Caution", "Curve left", "Curve right",
    "Double curve", "Bumpy road", "Slippery", "Road narrows", "Road work",
    "Signals", "Pedestrians", "Children", "Bicycles", "Ice/snow",
    "Wild animals", "End limits", "Turn right", "Turn left", "Ahead only",
    "Straight/right", "Straight/left", "Keep right", "Keep left",
    "Roundabout", "End no pass", "End no pass >3.5t"
]

DATA_DIR = "data"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────────
model = BaselineCNN().to(device)
model.load_state_dict(torch.load("experiments/baseline_cnn/best_model.pth", map_location=device, weights_only=True))
model.eval()

# ── Load test set ─────────────────────────────────────────────────────────────
df_test = pd.read_csv(f"{DATA_DIR}/Test.csv")
df_test["Path"] = df_test["Path"].str.replace("Test/", "test/", regex=False)
test_dataset = GTSRBDataset(df_test, DATA_DIR, get_val_transforms(32))
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# ── Collect all predictions ───────────────────────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── 1. Per-class report ───────────────────────────────────────────────────────
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ── 2. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(18, 16))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix — Test Set", color="white", fontsize=14, pad=14)
ax.set_xlabel("Predicted", color="#94a3b8", fontsize=11)
ax.set_ylabel("True", color="#94a3b8", fontsize=11)
ax.tick_params(colors="#64748b", labelsize=7)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()

# ── 3. Misclassified samples ──────────────────────────────────────────────────
wrong_idx = np.where(all_preds != all_labels)[0]
samples   = wrong_idx[:16]  # show first 16 mistakes

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(f"Misclassified Samples — {len(wrong_idx)} total errors", color="white", fontsize=13)

for ax, idx in zip(axes.flat, samples):
    row = df_test.iloc[idx]
    img = Image.open(f"{DATA_DIR}/{row['Path']}").convert("RGB")
    img = img.crop((row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]))
    img = img.resize((64, 64), Image.LANCZOS)
    ax.imshow(np.array(img))
    ax.set_title(
        f"True:  {CLASS_NAMES[all_labels[idx]]}\nPred: {CLASS_NAMES[all_preds[idx]]}",
        color="#ef4444", fontsize=6.5
    )
    ax.axis("off")

plt.tight_layout()
# plt.savefig("misclassified.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()