import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.transforms import get_train_transforms, get_val_transforms


class GTSRBDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]

        # Load image
        img = Image.open(os.path.join(self.data_dir, row["Path"])).convert("RGB")

        # Crop to ROI — always before resize
        img = img.crop((row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]))

        if self.transform:
            img = self.transform(img)

        return img, int(row["ClassId"])


def get_dataloaders(data_dir, img_size=32, batch_size=64, val_split=0.2, seed=42):
    csv_path = os.path.join(data_dir, "Train.csv")
    df = pd.read_csv(csv_path)
    df["Path"] = df["Path"].str.replace("Train/", "train/", regex=False)

    # ── Track-based split (prevents data leakage) ────────────────────────────
    df["TrackId"] = df["Path"].apply(
        lambda p: "_".join(p.split("/")[-1].split("_")[:2])
    )
    unique_tracks = df["TrackId"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_tracks)

    split_idx    = int((1 - val_split) * len(unique_tracks))
    train_tracks = set(unique_tracks[:split_idx])
    val_tracks   = set(unique_tracks[split_idx:])

    train_df = df[df["TrackId"].isin(train_tracks)].reset_index(drop=True)
    val_df   = df[df["TrackId"].isin(val_tracks)].reset_index(drop=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = GTSRBDataset(train_df, data_dir, get_train_transforms(img_size))
    val_dataset   = GTSRBDataset(val_df,   data_dir, get_val_transforms(img_size))

    # ── WeightedRandomSampler (handles class imbalance) ──────────────────────
    class_counts   = np.bincount(train_df["ClassId"].values)
    class_weights  = 1.0 / class_counts
    sample_weights = class_weights[train_df["ClassId"].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,        # no shuffle=True when using sampler
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train: {len(train_df):,} images  |  Val: {len(val_df):,} images")
    print(f"Train tracks: {len(train_tracks)}  |  Val tracks: {len(val_tracks)}")

    return train_loader, val_loader