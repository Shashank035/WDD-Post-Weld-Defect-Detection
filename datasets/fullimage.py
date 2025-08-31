import cv2
import os
from ultralytics import YOLO

def detect_defects(model_path, image_dir):
    # Load the saved model
    model = YOLO(model_path)

    # Iterate through all images in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Load an image
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)

            # Perform inference
            results = model(img)

            # Print detection results
            for result in results[0].boxes.data:
                class_id = int(result[5])
                confidence = float(result[4])
                x1, y1, x2, y2 = map(int, result[:4])

                print(f"Image: {filename}, Class: {results[0].names[class_id]}, Confidence: {confidence:.2f}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")

                # Draw bounding boxes on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{results[0].names[class_id]} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Resize the image to a medium size (e.g., 800x600)
            resized_img = cv2.resize(img, (800, 600))

            # Display the resized annotated image
            cv2.imshow('Defect Detection', resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Example usage
model_path = 'C:/Users/vijay/OneDrive/Desktop/WDD/runs/detect/ir_defect_detection3/weights/best.pt'
image_dir = 'C:/Users/vijay/OneDrive/Desktop/WDD/datasets/ir_images/ir'
detect_defects(model_path, image_dir)
