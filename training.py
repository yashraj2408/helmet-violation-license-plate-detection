from ultralytics import YOLO

# yolo model creation
model = YOLO("yolo-weights/yolov8l.pt")
model.train(
    data="datasets/helmet_dataset.yaml",
    imgsz=320,
    batch=4,
    epochs=20,
    workers=0,
    device="mps"  # use 'cuda' for GPU or 'cpu' if no MPS support
)
