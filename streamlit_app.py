"""
🏍️ Real-Time Motorcycle Helmet Violation & Automatic Number Plate Recognition (ANPR)
Streamlit Cloud Live Web Application
"""

import os
import re
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import easyocr

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Helmet Violation & ANPR AI",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS -----------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-num {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-violation {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-ok {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .plate-card {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- MODEL CACHING -----------------
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_yolo():
    # If custom weights exist, use them; otherwise use standard YOLOv8n
    weights_path = "model/best.pt" if os.path.exists("model/best.pt") else "yolov8n.pt"
    return YOLO(weights_path)

@st.cache_resource(show_spinner="Initializing EasyOCR engine (first time takes ~30s)...")
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

# ----------------- DETECTION LOGIC -----------------
def detect_helmet_on_person(person_crop, strictness=0.55):
    """
    Analyzes upper 30% of detected person to evaluate helmet compliance
    Returns: (has_helmet: bool, score: float)
    """
    if person_crop is None or person_crop.size == 0:
        return False, 0.0

    h, w = person_crop.shape[:2]
    if h < 25 or w < 25:
        return False, 0.0

    # Top 30% (head region)
    head_region = person_crop[0:int(h * 0.3), :]
    if head_region.size == 0:
        return False, 0.0

    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

    # Dark colors
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    dark_ratio = np.sum(dark_mask > 0) / dark_mask.size

    # Bright colors
    lower_bright = np.array([0, 0, 200])
    upper_bright = np.array([180, 255, 255])
    bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
    bright_ratio = np.sum(bright_mask > 0) / bright_mask.size

    # Color uniformity
    gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    uniformity_score = 1.0 if std_dev < 30 else (30.0 / std_dev)

    # Edge analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    edge_score = 1.0 if edge_ratio > 0.15 else (edge_ratio / 0.15)

    color_score = max(dark_ratio, bright_ratio)
    helmet_score = color_score * uniformity_score * edge_score

    has_helmet = helmet_score > strictness
    return has_helmet, float(helmet_score)


def detect_license_plate_regions(vehicle_crop):
    """
    Extracts license plate candidate contours based on aspect ratio
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []

    plates = []
    h, w = vehicle_crop.shape[:2]
    if h < 30 or w < 30:
        return []

    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        x, y, rect_w, rect_h = cv2.boundingRect(contour)
        aspect_ratio = rect_w / float(rect_h) if rect_h > 0 else 0

        if (1.8 <= aspect_ratio <= 7.0 and 
            rect_w > 45 and rect_h > 12 and
            rect_w < w * 0.95 and rect_h < h * 0.45 and
            rect_w * rect_h > 400):
            ideal_ratio = 3.5
            diff = abs(aspect_ratio - ideal_ratio)
            conf = max(0.0, 1.0 - (diff / 3.0))
            plates.append((x, y, rect_w, rect_h, conf))

    plates.sort(key=lambda item: item[4], reverse=True)
    return plates[:2]


def ocr_license_plate(plate_crop, reader):
    """
    Performs OCR reading with multiple enhancement methods
    """
    if plate_crop is None or plate_crop.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 60:
        scale = 80 / max(h, 1)
        gray = cv2.resize(gray, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)

    results_all = []

    # Method 1: CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img1 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    try:
        results1 = reader.readtext(img1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1)
        results_all.extend(results1)
    except Exception:
        pass

    # Method 2: Otsu Binary
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img2 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    try:
        results2 = reader.readtext(img2, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=1)
        results_all.extend(results2)
    except Exception:
        pass

    if results_all:
        best = max(results_all, key=lambda x: x[2])
        cleaned = re.sub(r'[^A-Z0-9]', '', str(best[1]).upper())
        conf = float(best[2])
        if len(cleaned) >= 2 and conf >= 0.15:
            return cleaned, conf

    return None, 0.0


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/motorcycle.png", width=64)
    st.title("Settings & Controls")
    
    st.markdown("### Detection Parameters")
    conf_threshold = st.slider("YOLO Confidence Threshold", 0.15, 0.90, 0.35, 0.05)
    strictness = st.slider("Helmet Strictness Threshold", 0.30, 0.85, 0.55, 0.05,
                           help="Higher = stricter criteria for marking helmet as compliant")
    
    st.markdown("### Feature Toggles")
    enable_ocr = st.checkbox("Enable Number Plate OCR", value=True)
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.checkbox("Show Labels & Scores", value=True)
    
    st.markdown("---")
    st.caption("Engine: YOLOv8 + EasyOCR | Hosting: Streamlit Community Cloud")

# ----------------- MAIN UI -----------------
st.markdown('<div class="main-title">🏍️ Helmet Violation & License Plate Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated AI Traffic Safety Monitoring System with Real-Time YOLOv8 Detection & EasyOCR ANPR</div>', unsafe_allow_html=True)

# Tabs for input modes
tab_upload, tab_sample, tab_about = st.tabs(["📤 Upload Image", "🌟 Try Demo Samples", "ℹ️ About the Project"])

input_image = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload a traffic or rider photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

with tab_sample:
    st.write("Don't have an image ready? Select one of our built-in test images:")
    sample_col1, sample_col2 = st.columns(2)
    sample_choice = None
    
    if os.path.exists("sample_images/demo1.jpg"):
        with sample_col1:
            st.image("sample_images/demo1.jpg", caption="Sample 1 (Traffic Frame 1)", use_container_width=True)
            if st.button("Analyze Sample 1", key="btn_s1"):
                input_image = Image.open("sample_images/demo1.jpg").convert("RGB")
                
    if os.path.exists("sample_images/demo2.jpg"):
        with sample_col2:
            st.image("sample_images/demo2.jpg", caption="Sample 2 (Traffic Frame 2)", use_container_width=True)
            if st.button("Analyze Sample 2", key="btn_s2"):
                input_image = Image.open("sample_images/demo2.jpg").convert("RGB")

with tab_about:
    st.markdown("""
    ### 🎯 System Features
    - **YOLOv8 Deep Learning**: Real-time object localization for riders, motorcycles, and vehicles.
    - **Intelligent Helmet Classification**: Multi-criteria analysis examining head region HSV color space, edge smoothness, and color uniformity.
    - **Automatic Number Plate Recognition (ANPR)**: Automatic candidate contour filtering with EasyOCR alphanumeric extraction.
    - **Cloud Ready**: Configured for 24/7 serverless deployment on Streamlit Community Cloud.
    """)

# ----------------- INFERENCE PIPELINE -----------------
if input_image is not None:
    st.markdown("---")
    
    yolo_model = load_yolo()
    reader = load_ocr() if enable_ocr else None
    
    # Convert PIL to cv2 BGR
    img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    orig_display = np.array(input_image)
    
    with st.spinner("Processing image through AI vision pipeline..."):
        results = yolo_model.predict(source=img_bgr, conf=conf_threshold, verbose=False)
        
        persons_detected = 0
        violations_count = 0
        compliant_count = 0
        detected_plates_list = []
        detection_details = []
        
        annotated_bgr = img_bgr.copy()
        h_img, w_img = annotated_bgr.shape[:2]
        
        # Vehicle classes in COCO: motorcycle(3), car(2), bus(5), truck(7)
        VEHICLE_CLASSES = {"motorcycle", "car", "bus", "truck"}
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = yolo_model.names.get(cls_id, str(cls_id)).lower()
                conf = float(box.conf[0])
                
                # Check for Person
                if label == "person":
                    persons_detected += 1
                    person_crop = img_bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                    
                    has_helmet, helmet_score = detect_helmet_on_person(person_crop, strictness=strictness)
                    
                    if has_helmet:
                        compliant_count += 1
                        box_color = (0, 200, 0) # Green BGR
                        status_str = f"Helmet Compliant ({helmet_score:.2f})"
                        tag = "Compliant"
                    else:
                        violations_count += 1
                        box_color = (0, 0, 255) # Red BGR
                        status_str = f"NO HELMET VIOLATION ({conf:.2f})"
                        tag = "Violation"
                        
                    detection_details.append({
                        "Type": "Person / Rider",
                        "Status": tag,
                        "Confidence": f"{conf:.2f}",
                        "Helmet Score": f"{helmet_score:.2f}",
                        "Box": f"[{x1}, {y1}, {x2}, {y2}]"
                    })
                    
                    if show_boxes:
                        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), box_color, 3)
                        if show_labels:
                            cv2.putText(annotated_bgr, status_str, (x1, max(20, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                # Check for Vehicle & License Plate detection
                elif label in VEHICLE_CLASSES or "plate" in label:
                    if enable_ocr and reader is not None:
                        veh_crop = img_bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                        plates = detect_license_plate_regions(veh_crop)
                        
                        for (px, py, pw, ph, pconf) in plates:
                            abs_x = x1 + px
                            abs_y = y1 + py
                            plate_crop = veh_crop[py:py+ph, px:px+pw]
                            
                            plate_text, ocr_conf = ocr_license_plate(plate_crop, reader)
                            
                            if plate_text:
                                detected_plates_list.append({
                                    "text": plate_text,
                                    "confidence": ocr_conf,
                                    "crop": cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                                })
                                
                                if show_boxes:
                                    cv2.rectangle(annotated_bgr, (abs_x, abs_y), (abs_x + pw, abs_y + ph), (255, 255, 0), 2)
                                    if show_labels:
                                        cv2.putText(annotated_bgr, f"Plate: {plate_text}", (abs_x, max(15, abs_y - 6)),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                                break
                                
                    if show_boxes:
                        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (200, 150, 0), 1)
                        if show_labels:
                            cv2.putText(annotated_bgr, f"{label} {conf:.2f}", (x1, max(15, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 0), 1)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # ----------------- METRICS DASHBOARD -----------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#2563EB;">{persons_detected}</div><div class="metric-label">Persons Detected</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#DC2626;">{violations_count}</div><div class="metric-label">Helmet Violations</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#16A34A;">{compliant_count}</div><div class="metric-label">Helmet Compliant</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#D97706;">{len(detected_plates_list)}</div><div class="metric-label">Plates Recognized</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # ----------------- SIDE-BY-SIDE VISUALS -----------------
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📷 Original Image")
        st.image(orig_display, use_container_width=True)
        
    with col_right:
        st.subheader("🔍 AI Detection Result")
        st.image(annotated_rgb, use_container_width=True)
        
        # Download analyzed image button
        res_pil = Image.fromarray(annotated_rgb)
        from io import BytesIO
        buf = BytesIO()
        res_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Analyzed Image",
            data=buf.getvalue(),
            file_name="helmet_anpr_result.png",
            mime="image/png",
            use_container_width=True
        )

    # ----------------- DETECTED LICENSE PLATES -----------------
    if detected_plates_list:
        st.markdown("### 🪪 Detected License Plates")
        plate_cols = st.columns(min(4, len(detected_plates_list)))
        for i, plate in enumerate(detected_plates_list):
            with plate_cols[i % len(plate_cols)]:
                st.markdown(f"""
                <div class="plate-card">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #166534; letter-spacing: 0.05em;">{plate['text']}</div>
                    <div style="font-size: 0.8rem; color: #4B5563;">OCR Confidence: {plate['confidence']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(plate['crop'], caption="Plate Crop", width=160)

    # ----------------- VIOLATION BREAKDOWN TABLE -----------------
    if detection_details:
        with st.expander("📊 View Detailed Detections Table", expanded=(violations_count > 0)):
            st.dataframe(detection_details, use_container_width=True)
            if violations_count > 0:
                st.error(f"⚠️ Alert: {violations_count} helmet violation(s) flagged in this frame. Automatic fine / citation dispatch recommended.")
            else:
                st.success("✅ All detected riders in this frame are wearing helmets.")

else:
    st.info("👆 Please upload an image or select one of the built-in demo samples above to begin analysis.")
