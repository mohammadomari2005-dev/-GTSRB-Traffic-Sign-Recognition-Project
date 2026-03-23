import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from src.model import BaselineCNN
from src.transforms import get_val_transforms

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


def load_model(checkpoint_path, device):
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model


def classify_crop(model, crop_rgb, transform, device):
    img    = Image.fromarray(crop_rgb)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob   = torch.softmax(output, dim=1)
        conf   = prob.max().item()
        label  = prob.argmax().item()
    return CLASS_NAMES[label], conf


def run_video(video_path, output_path, checkpoint_path, conf_threshold=0.15):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(checkpoint_path, device)
    detector = YOLO("experiments/gtsrb_detector_v2/weights/best.pt")
    transform = get_val_transforms(32)

    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Processing {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames on {device}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = detector(frame, verbose=False, conf=conf_threshold)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1

            # Skip tiny detections
            if bw < 20 or bh < 20:
                continue

            # crop       = frame_rgb[y1:y2, x1:x2]

            pad = 5
            y1p = max(0, y1 - pad)
            y2p = min(frame.shape[0], y2 + pad)
            x1p = max(0, x1 - pad)
            x2p = min(frame.shape[1], x2 + pad)
            crop = frame_rgb[y1p:y2p, x1p:x2p]

            name, conf = classify_crop(model, crop, transform, device)
            label      = f"{name} {conf:.0%}"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Done — saved to {output_path}")


if __name__ == "__main__":
    run_video(
        video_path="data/Untitled design.mp4",
        output_path="data/output_video.mp4",
        checkpoint_path="experiments/baseline_cnn/best_model.pth",
    )