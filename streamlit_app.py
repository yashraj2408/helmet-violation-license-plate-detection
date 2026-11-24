"""
🏍️ Real-Time Motorcycle Helmet Violation & Automatic Number Plate Recognition (ANPR)
Streamlit Cloud Live Web Application powered by Dedicated YOLOv8 Models & EasyOCR
"""

import os
import re
import urllib.request
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
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-num {
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .plate-card {
        background: linear-gradient(135deg, #FEFCE8 0%, #FEF08A 100%);
        border: 2px solid #EAB308;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    .plate-text {
        font-family: 'Courier New', monospace;
        font-size: 1.3rem;
        font-weight: 800;
        color: #854D0E;
        letter-spacing: 0.1em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- MODEL MANAGEMENT -----------------
MODEL_DIR = "weights"
HELMET_MODEL_PATH = os.path.join(MODEL_DIR, "helmet_yolov8n.pt")
PLATE_MODEL_PATH = os.path.join(MODEL_DIR, "license_plate_yolov8n.pt")

HELMET_URL = "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/best.pt"
PLATE_URL = "https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt"

def ensure_model_file(path, url, name):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) < 1000000:
        with st.spinner(f"Downloading {name} model weights (~6 MB)..."):
            urllib.request.urlretrieve(url, path)

@st.cache_resource(show_spinner="Loading AI Vision Models...")
def load_models():
    ensure_model_file(HELMET_MODEL_PATH, HELMET_URL, "Helmet Detection")
    ensure_model_file(PLATE_MODEL_PATH, PLATE_URL, "License Plate Detection")
    
    helmet_model = YOLO(HELMET_MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    return helmet_model, plate_model, ocr_reader

# ----------------- OCR PREPROCESSING -----------------
def clean_plate_text(raw_text):
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return cleaned

def read_license_plate(plate_crop, reader):
    if plate_crop is None or plate_crop.size == 0:
        return "", 0.0

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 50:
        scale = 80 / max(h, 1)
        gray = cv2.resize(gray, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)

    results_all = []
    
    # 1. Raw resized
    try:
        r1 = reader.readtext(gray, detail=1)
        results_all.extend(r1)
    except Exception:
        pass

    # 2. Enhanced CLAHE contrast
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        r2 = reader.readtext(enhanced, detail=1)
        results_all.extend(r2)
    except Exception:
        pass

    if not results_all:
        return "", 0.0

    # Combine detected text snippets ordered by confidence
    valid_texts = []
    confidences = []
    for bbox, text, conf in results_all:
        cl = clean_plate_text(text)
        if len(cl) >= 2 and conf > 0.15:
            valid_texts.append(cl)
            confidences.append(float(conf))

    if valid_texts:
        # Join unique detected words or pick best
        final_text = " ".join(dict.fromkeys(valid_texts))
        avg_conf = sum(confidences) / len(confidences)
        return final_text, avg_conf

    return "", 0.0

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/motorcycle.png", width=64)
    st.title("Settings & Controls")
    
    st.markdown("### Model Sensitivity")
    helmet_conf = st.slider("Helmet Detection Confidence", 0.15, 0.85, 0.25, 0.05,
                            help="Threshold for detecting riders with/without helmet")
    plate_conf = st.slider("License Plate Confidence", 0.10, 0.85, 0.20, 0.05,
                           help="Threshold for detecting license plates")
    
    st.markdown("### Display Options")
    show_boxes = st.checkbox("Show Bounding Boxes", value=True)
    show_labels = st.checkbox("Show Labels & Scores", value=True)
    enable_ocr = st.checkbox("Enable Number Plate OCR", value=True)
    
    st.markdown("---")
    st.caption("AI Models: YOLOv8n-Helmet + YOLOv8n-Plate + EasyOCR")

# ----------------- MAIN UI -----------------
st.markdown('<div class="main-title">🏍️ Real-Time Helmet Violation & License Plate Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated AI Traffic Safety Monitoring powered by dedicated YOLOv8 Computer Vision Models</div>', unsafe_allow_html=True)

tab_upload, tab_sample, tab_about = st.tabs(["📤 Upload Image", "🌟 Try Built-in Demo Samples", "ℹ️ Model Details"])

input_image = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload a traffic, bike, or street photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

with tab_sample:
    st.write("Click any sample image below to run immediate detection:")
    c1, c2, c3 = st.columns(3)
    
    if os.path.exists("sample_images/demo1.jpg"):
        with c1:
            st.image("sample_images/demo1.jpg", caption="Sample 1: Bike Stunt (Violation)", use_container_width=True)
            if st.button("⚡ Analyze Sample 1", key="btn_s1", use_container_width=True):
                input_image = Image.open("sample_images/demo1.jpg").convert("RGB")

    if os.path.exists("sample_images/demo2.jpg"):
        with c2:
            st.image("sample_images/demo2.jpg", caption="Sample 2: Motion Frame", use_container_width=True)
            if st.button("⚡ Analyze Sample 2", key="btn_s2", use_container_width=True):
                input_image = Image.open("sample_images/demo2.jpg").convert("RGB")

    if os.path.exists("sample_images/demo3_traffic_bike.jpg"):
        with c3:
            st.image("sample_images/demo3_traffic_bike.jpg", caption="Sample 3: Real Traffic (2 Riders + Plate)", use_container_width=True)
            if st.button("⚡ Analyze Sample 3", key="btn_s3", use_container_width=True):
                input_image = Image.open("sample_images/demo3_traffic_bike.jpg").convert("RGB")

with tab_about:
    st.markdown("""
    ### 🚀 Specialized AI Architecture
    1. **Helmet Classifier (YOLOv8n)**: Dedicated neural network trained directly on helmet vs. no-helmet datasets (`iam-tsr/yolov8n-helmet-detection`).
    2. **License Plate Localizer (YOLOv8n)**: Dedicated neural network trained to detect motor vehicle number plates (`joker5914/yolov8n-license-plate`).
    3. **Optical Character Recognition (EasyOCR)**: High-precision text extraction from cropped plates with CLAHE contrast enhancement.
    """)

# ----------------- INFERENCE PIPELINE -----------------
if input_image is not None:
    st.markdown("---")
    
    helmet_model, plate_model, ocr_reader = load_models()
    
    img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    orig_display = np.array(input_image)
    annotated_bgr = img_bgr.copy()
    
    with st.spinner("Analyzing image with neural networks..."):
        # 1. Run Helmet Detection
        helmet_results = helmet_model.predict(source=img_bgr, conf=helmet_conf, verbose=False)
        
        violations_count = 0
        compliant_count = 0
        records = []
        
        for r in helmet_results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                label = helmet_model.names.get(cls_id, str(cls_id))
                conf = float(b.conf[0])
                
                is_violation = ("without" in label.lower() or "no" in label.lower())
                
                if is_violation:
                    violations_count += 1
                    color = (0, 0, 255) # Red BGR
                    tag = "🚨 NO HELMET (VIOLATION)"
                else:
                    compliant_count += 1
                    color = (0, 220, 0) # Green BGR
                    tag = "✅ WITH HELMET (COMPLIANT)"
                    
                records.append({
                    "Detection": "Helmet Status",
                    "Classification": tag,
                    "Confidence": f"{conf * 100:.1f}%",
                    "Coordinates": f"[{x1}, {y1}, {x2}, {y2}]"
                })
                
                if show_boxes:
                    cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)
                    if show_labels:
                        text_label = f"{tag}: {conf:.2f}"
                        cv2.putText(annotated_bgr, text_label, (x1, max(22, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 2. Run License Plate Detection
        plate_results = plate_model.predict(source=img_bgr, conf=plate_conf, verbose=False)
        detected_plates = []
        
        for r in plate_results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                p_conf = float(b.conf[0])
                
                h_img, w_img = img_bgr.shape[:2]
                plate_crop = img_bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                
                plate_text = ""
                ocr_conf = 0.0
                
                if enable_ocr and plate_crop.size > 0:
                    plate_text, ocr_conf = read_license_plate(plate_crop, ocr_reader)
                    
                detected_plates.append({
                    "text": plate_text if plate_text else "PLATE DETECTED",
                    "plate_conf": p_conf,
                    "ocr_conf": ocr_conf,
                    "crop": cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB) if plate_crop.size > 0 else None
                })
                
                records.append({
                    "Detection": "License Plate",
                    "Classification": plate_text if plate_text else "Detected Plate",
                    "Confidence": f"{p_conf * 100:.1f}%",
                    "Coordinates": f"[{x1}, {y1}, {x2}, {y2}]"
                })
                
                if show_boxes:
                    cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2) # Cyan
                    if show_labels:
                        disp_str = f"Plate: {plate_text}" if plate_text else f"Plate ({p_conf:.2f})"
                        cv2.putText(annotated_bgr, disp_str, (x1, max(18, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # ----------------- METRICS ROW -----------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#2563EB;">{violations_count + compliant_count}</div><div class="metric-label">Riders Analyzed</div></div>', unsafe_allow_html=True)
    with m2:
        v_color = "#DC2626" if violations_count > 0 else "#64748B"
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:{v_color};">{violations_count}</div><div class="metric-label">Helmet Violations</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#16A34A;">{compliant_count}</div><div class="metric-label">Wearing Helmet</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#D97706;">{len(detected_plates)}</div><div class="metric-label">Plates Localized</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # ----------------- VISUAL COMPARISON -----------------
    c_orig, c_pred = st.columns(2)
    with c_orig:
        st.subheader("📷 Original Input")
        st.image(orig_display, use_container_width=True)
    with c_pred:
        st.subheader("🔍 AI Detections (Helmets & Number Plates)")
        st.image(annotated_rgb, use_container_width=True)
        
        # Download button
        res_pil = Image.fromarray(annotated_rgb)
        from io import BytesIO
        buf = BytesIO()
        res_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Analyzed Image",
            data=buf.getvalue(),
            file_name="detection_result.png",
            mime="image/png",
            use_container_width=True
        )

    # ----------------- LICENSE PLATES CARDS -----------------
    if detected_plates:
        st.markdown("### 🪪 Recognized License Plates")
        plate_cols = st.columns(max(1, min(4, len(detected_plates))))
        for idx, p_info in enumerate(detected_plates):
            with plate_cols[idx % len(plate_cols)]:
                st.markdown(f"""
                <div class="plate-card">
                    <div class="plate-text">{p_info['text']}</div>
                    <div style="font-size:0.75rem; color:#713F12; margin-top:4px;">
                        Plate Conf: {p_info['plate_conf']*100:.1f}% | OCR: {p_info['ocr_conf']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if p_info['crop'] is not None:
                    st.image(p_info['crop'], caption="Cropped Plate", width=180)

    # ----------------- AUDIT REPORT -----------------
    if records:
        with st.expander("📊 View Detailed Vision Audit Log", expanded=(violations_count > 0)):
            st.dataframe(records, use_container_width=True)
            if violations_count > 0:
                st.error(f"⚠️ ENFORCEMENT ALERT: Flagged {violations_count} helmet violation(s). Vehicle plate registration logged for automated citation.")
            else:
                st.success("✅ Traffic Compliance: All riders are wearing helmets.")
else:
    st.info("👆 Upload an image or select one of the built-in sample images above to see the new AI models in action.")
