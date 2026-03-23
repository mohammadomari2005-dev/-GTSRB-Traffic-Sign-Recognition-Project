from ultralytics import YOLO

def train():
    model = YOLO("yolov8s.pt")   # start from pretrained weights

    results = model.train(
    data="detection/german_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="gtsrb_german_detector",
    project="experiments",
    patience=5,
    device="cuda",
)

    print(f"\n✅ Training complete")
    print(f"Best model saved to: experiments/gtsrb_detector_v2/weights/best.pt")
if __name__ == "__main__":
    train()