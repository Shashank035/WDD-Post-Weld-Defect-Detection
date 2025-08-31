# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import cv2
# import torch
# import numpy as np
# import os

# # Load YOLO model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # You can change 'yolov5s' to other sizes like 'yolov5m', 'yolov5l', 'yolov5x'

# # Set device to CUDA if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device).eval()

# def detect_defects(img_path, output_folder):
#     try:
#         # Load the image
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

#         # Perform inference
#         results = model(img)

#         # Parse detections
#         predictions = results.xyxy[0].cpu().numpy()  # Get the bounding box coordinates
#         labels = results.names  # Get the label names

#         # Draw bounding boxes on the image
#         for *xyxy, conf, cls in predictions:
#             x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to integers
#             label = f"{labels[int(cls)]} {conf:.2f}"  # Create label with class name and confidence
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#         return

#     except Exception as e:
#         print(f"Error detecting defects in image: {e}")
#         return None

# if __name__ == "__main__":
#     # Dataset path
#     dataset_path = r'C:\Users\vijay\OneDrive\Desktop\WDD\datasets\ir_images\ir'
#     output_path = r'C:\Users\vijay\OneDrive\Desktop\WDD\output\detected_defects'

#     # Make output directory if not exists
#     os.makedirs(output_path, exist_ok=True)

#     # Iterate over the images
#     for filename in os.listdir(dataset_path):
#         img_path = os.path.join(dataset_path, filename)

#         # Detect defects and display the image
#         detect_defects(img_path, output_path)
