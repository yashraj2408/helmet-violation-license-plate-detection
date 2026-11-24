from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import cv2
import os
import uuid
import easyocr

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8 model (fallback to yolov8n if best.pt not present)
model_path = "model/best.pt" if os.path.exists("model/best.pt") else "yolov8n.pt"
model = YOLO(model_path)

# Initialize OCR reader
reader = easyocr.Reader(['en'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    # Save uploaded file
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run YOLOv8 prediction
    results = model.predict(source=filepath, conf=0.4, save=False)
    img = cv2.imread(filepath)

    helmet_confidences = []

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            # Draw bounding box
            color = (0, 255, 0) if "helmet" in label.lower() else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Collect helmet confidence
            if "helmet" in label.lower():
                helmet_confidences.append(conf)

    # If any helmet detected, show the highest confidence on the image
    if helmet_confidences:
        max_conf = max(helmet_confidences)
        conf_text = f"Helmet Confidence: {max_conf:.2f}"
        cv2.putText(img, conf_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)

    # Number plate OCR (internal only, not displayed)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            if "plate" in label.lower() or "number" in label.lower():
                plate_crop = img[y1:y2, x1:x2]
                _ = reader.readtext(plate_crop, detail=0)

    # Save output image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "pred_" + filename)
    cv2.imwrite(output_path, img)

    # Render images side by side
    return render_template('index.html',
                           uploaded_image=url_for('static', filename='uploads/' + filename),
                           output_image=url_for('static', filename='uploads/' + "pred_" + filename))

if __name__ == "__main__":
    app.run(debug=True)
