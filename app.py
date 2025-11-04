import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import DetrForObjectDetection, DetrImageProcessor
from ensemble_boxes import weighted_boxes_fusion

# ============================================================
# 🧠 LOAD MODELS
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Device in use: {device.upper()}")

@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    detr_model = 
DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    processor = 
DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    return yolo_model, detr_model, processor

yolo_model, detr_model, processor = load_models()

# ============================================================
# ⚙️ DETECTION HELPERS
# ============================================================
def infer_yolo(image, conf_thr=0.25):
    results = yolo_model(image, conf=conf_thr, verbose=False)[0]
    boxes, scores = [], []
    h, w = image.shape[:2]
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf[0])
        boxes.append([xyxy[0]/w, xyxy[1]/h, xyxy[2]/w, xyxy[3]/h])
        scores.append(conf)
    return boxes, scores

def infer_detr(image, conf_thr=0.25):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detr_model(**inputs)
    target_sizes = torch.tensor([[image.shape[0], 
image.shape[1]]]).to(device)
    results = processor.post_process_object_detection(outputs, 
target_sizes=target_sizes)[0]
    boxes, scores = [], []
    for s, b in zip(results["scores"], results["boxes"]):
        if s > conf_thr:
            x1, y1, x2, y2 = b.cpu().numpy().tolist()
            boxes.append([x1/image.shape[1], y1/image.shape[0], 
x2/image.shape[1], y2/image.shape[0]])
            scores.append(float(s))
    return boxes, scores

def infer_hybrid(image, conf_thr=0.25, iou_thr=0.5):
    yolo_boxes, yolo_scores = infer_yolo(image, conf_thr)
    detr_boxes, detr_scores = infer_detr(image, conf_thr)
    if not yolo_boxes and not detr_boxes:
        return [], []
    boxes, scores, _ = weighted_boxes_fusion(
        [yolo_boxes, detr_boxes],
        [yolo_scores, detr_scores],
        [[0]*len(yolo_boxes), [0]*len(detr_boxes)],
        weights=[3, 1],
        iou_thr=iou_thr
    )
    h, w = image.shape[:2]
    boxes = (boxes * np.array([w, h, w, h])).astype(int)
    return boxes, scores

# ============================================================
# 🖼️ STREAMLIT UI
# ============================================================
st.title("🔮 YOLO + DETR Hybrid Object Detection")
st.markdown("Upload an image and choose a model (YOLO, DETR, or HYBRID).")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", 
"jpeg", "png"])
model_choice = st.selectbox("Choose model:", ["YOLOv8", "DETR", "HYBRID"])
conf_thr = st.slider("Confidence Threshold", 0.1, 0.9, 0.4)
iou_thr = st.slider("IoU Threshold (for hybrid)", 0.1, 0.9, 0.5)

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🚀 Run Detection"):
        with st.spinner(f"Running {model_choice} detection..."):
            if model_choice == "YOLOv8":
                boxes, scores = infer_yolo(image_bgr, conf_thr)
                h, w = image.shape[:2]
                boxes = (np.array(boxes) * np.array([w, h, w, 
h])).astype(int)
            elif model_choice == "DETR":
                boxes, scores = infer_detr(image_bgr, conf_thr)
                h, w = image.shape[:2]
                boxes = (np.array(boxes) * np.array([w, h, w, 
h])).astype(int)
            else:
                boxes, scores = infer_hybrid(image_bgr, conf_thr, iou_thr)

            # Draw boxes
            for (x1, y1, x2, y2), score in zip(boxes, scores):
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 
2)
                cv2.putText(image_bgr, f"{score:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                     caption=f"{model_choice} Detection Results", 
use_column_width=True)
            st.success(f"✅ {len(boxes)} objects detected!")


