from ultralytics import YOLO
import math
import cv2
import cvzone
import torch
from image_to_text import predict_number_plate
from paddleocr import PaddleOCR

# -------------------- CONFIG --------------------
VIDEO_PATH = "/Users/yashraj/sem7/capstone/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main/videos/22.mp4"
MODEL_PATH = "best.pt"  # Path to your trained YOLOv8 model
OUTPUT_PATH = "output.mp4"

# Use MPS for Apple Silicon (or change to 'cuda' for NVIDIA GPU / 'cpu' otherwise)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Class labels (must match your training)
classNames = ["with helmet", "without helmet", "rider", "number plate"]

# -------------------- INITIALIZATION --------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Could not open video: {VIDEO_PATH}")

# Get frame details
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Load YOLO model
model = YOLO(MODEL_PATH)
model.to(device)

# Load OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

print("✅ Models loaded successfully. Starting detection...")

# -------------------- DETECTION LOOP --------------------
while True:
    success, img = cap.read()
    if not success:
        print("✅ Video processing complete.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, stream=True, device=device)

    li = {}
    rider_boxes = []

    for r in results:
        boxes = r.boxes
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device),
                               confidences.unsqueeze(1).to(device),
                               classes.unsqueeze(1).to(device)), 1)

        # Sort by class index to process in order
        new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]

        for box in new_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            conf = float(box[4])
            cls = int(box[5])

            # Filter relevant detections
            if (classNames[cls] == "without helmet" and conf >= 0.5) or \
               (classNames[cls] == "rider" and conf >= 0.45) or \
               (classNames[cls] == "number plate" and conf >= 0.5):

                if classNames[cls] == "rider":
                    rider_boxes.append((x1, y1, x2, y2))

                for j, rider in enumerate(rider_boxes):
                    if (x1 >= rider[0] and y1 >= rider[1] and x2 <= rider[2] and y2 <= rider[3]):
                        # Draw box
                        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                        cvzone.putTextRect(img, f"{classNames[cls].upper()}",
                                           (x1 + 10, y1 - 10), scale=1.5,
                                           offset=10, thickness=2,
                                           colorT=(39, 40, 41), colorR=(248, 222, 34))

                        li.setdefault(f"rider{j}", []).append(classNames[cls])

                        # If number plate detected, crop
                        if classNames[cls] == "number plate":
                            crop = img[y1:y2, x1:x2]

                        # Once all 3 detections (helmet, rider, number plate) found for rider
                        if len(set(li[f"rider{j}"])) == 3:
                            try:
                                vechicle_number, conf = predict_number_plate(crop, ocr)
                                if vechicle_number and conf:
                                    cvzone.putTextRect(
                                        img,
                                        f"{vechicle_number} {round(conf * 100, 2)}%",
                                        (x1, y1 - 50), scale=1.5,
                                        offset=10, thickness=2,
                                        colorT=(39, 40, 41),
                                        colorR=(105, 255, 255)
                                    )
                            except Exception as e:
                                print("⚠️ OCR Error:", e)

    # Write frame to output and display
    output.write(img)
    cv2.imshow("Helmet & Number Plate Detection", img)

    # Exit when 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- CLEANUP --------------------
cap.release()
output.release()
cv2.destroyAllWindows()
print("🎬 Output saved as:", OUTPUT_PATH)
