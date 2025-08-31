import streamlit as st
import cv2
import os
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model_path = './runs/detect/ir_defect_detection3/weights/best.pt'

model = YOLO(model_path)

def detect_defects(image):
    # Perform inference
    results = model(image)

    # Extract detection results
    detections = []
    for result in results[0].boxes.data:
        class_id = int(result[5])
        confidence = float(result[4])
        x1, y1, x2, y2 = map(int, result[:4])

        detections.append({
            'class': results[0].names[class_id],
            'confidence': confidence,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        })

    return detections

def main():
    st.title("Defect Detection App")
    uploaded_files = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            detections = detect_defects(image)

            # Display the original image
            st.header("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Display detection results
            if detections:
                st.header("Detected Defects:")
                for detection in detections:
                    st.write(f"- **Class:** {detection['class']}, **Confidence:** {detection['confidence']:.2f}")
            else:
                st.header("No Defects Found")

if __name__ == "__main__":
    main()
