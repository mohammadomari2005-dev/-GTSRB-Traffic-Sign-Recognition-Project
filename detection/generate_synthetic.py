# detection/generate_synthetic.py
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import random

DATA_DIR    = "data"
BG_DIR      = "detection/backgrounds"
OUT_IMG_DIR = "detection/german_dataset/train/images"
OUT_LBL_DIR = "detection/german_dataset/train/labels"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

df = pd.read_csv(f"{DATA_DIR}/Train.csv")
df["Path"] = df["Path"].str.replace("Train/", "train/", regex=False)
backgrounds = [f"{BG_DIR}/{f}" for f in os.listdir(BG_DIR) if f.endswith(".jpg")]

def paste_sign(bg_img, sign_img, x, y):
    sh, sw = sign_img.shape[:2]
    bh, bw = bg_img.shape[:2]
    if x + sw > bw or y + sh > bh:
        return bg_img, None
    bg_img[y:y+sh, x:x+sw] = sign_img
    # YOLO format: class cx cy w h (normalized)
    cx = (x + sw / 2) / bw
    cy = (y + sh / 2) / bh
    w  = sw / bw
    h  = sh / bh
    return bg_img, (cx, cy, w, h)

N = 5000  # number of synthetic images to generate
for i in range(N):
    # Pick random background
    bg_path = random.choice(backgrounds)
    bg = cv2.imread(bg_path)
    bh, bw = bg.shape[:2]

    # Pick random sign
    row = df.sample(1).iloc[0]
    class_id = int(row["ClassId"])
    img = Image.open(f"{DATA_DIR}/{row['Path']}").convert("RGB")
    img = img.crop((row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]))

    # Random scale between 5% and 15% of frame width
    scale = random.uniform(0.05, 0.15)
    new_w = int(bw * scale)
    new_h = int(new_w * img.height / img.width)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    sign = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Random position
    x = random.randint(0, max(0, bw - new_w))
    y = random.randint(0, max(0, bh - new_h))

    result, bbox = paste_sign(bg.copy(), sign, x, y)
    if bbox is None:
        continue

    # Save image and label
    img_name = f"synthetic_{i:05d}.jpg"
    cv2.imwrite(f"{OUT_IMG_DIR}/{img_name}", result)
    with open(f"{OUT_LBL_DIR}/synthetic_{i:05d}.txt", "w") as f:
        f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    if i % 500 == 0:
        print(f"Generated {i}/{N} images")

print("✅ Done generating synthetic dataset")