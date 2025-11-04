import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import DetrForObjectDetection, DetrImageProcessor
from ensemble_boxes import weighted_boxes_fusion

# --------------------------------------------------
# Load Models
# --------------------------------------------------
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov8n.pt")  # lightweight for Streamlit
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model.eval()
    return yolo_model, detr_model, processor

yolo_model, detr_model, processor = load_models()
device = "cuda" if torch.cuda.is_available() else "cpu"
detr_model.to(device)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def run_yolo(img, conf_thr=0.25):
    results = yolo_model(img, verbose=False)[0]
    h, w = img.shape[:2]
    boxes, scores = [], []
    for box in results.boxes:
        xyxy, conf = box.xyxy[0].tolist(), box.conf[0].item()
        if conf >= conf_thr:
            boxes.append([xyxy[0]/w, xyxy[1]/h, xyxy[2]/w, xyxy[3]/h])
            scores.append(conf)
    return boxes, scores

def run_detr(img, conf_thr=0.25):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detr_model(**inputs)
    target_sizes = torch.tensor([[img.shape[0], img.shape[1]]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    boxes, scores = [], []
    for s, b in zip(results["scores"], results["boxes"]):
        if s > conf_thr:
            x1, y1, x2, y2 = b.tolist()
            boxes.append([x1/img.shape[1], y1/img.shape[0], x2/img.shape[1], y2/img.shape[0]])
            scores.append(float(s))
    return boxes, scores

def hybrid_fusion(yolo_boxes, yolo_scores, detr_boxes, detr_scores, iou_thr=0.5):
    if not yolo_boxes and not detr_boxes:
        return []
    boxes, scores, labels = weighted_boxes_fusion(
        [yolo_boxes, detr_boxes],
        [yolo_scores, detr_scores],
        [[0]*len(yolo_boxes), [0]*len(detr_boxes)],
        weights=[3, 1],
        iou_thr=iou_thr
    )
    return boxes, scores

def draw_boxes(img, boxes, scores):
    output = img.copy()
    h, w = output.shape[:2]
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [b[0]*w, b[1]*h, b[2]*w, b[3]*h])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"{s:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🧠 Hybrid Object Detector (YOLO + DETR)")
st.write("Upload an image to detect objects using both YOLOv8 and DETR with weighted fusion.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running hybrid detection..."):
        yolo_boxes, yolo_scores = run_yolo(img)
        detr_boxes, detr_scores = run_detr(img)
        fused_boxes, fused_scores = hybrid_fusion(yolo_boxes, yolo_scores, detr_boxes, detr_scores)
        output = draw_boxes(img, fused_boxes, fused_scores)

    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)
    st.success(f"✅ Detected {len(fused_boxes)} objects")

