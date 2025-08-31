import cv2
import os
from ultralytics import YOLO

def detect_defects(model_path, image_dir, output_dir):
    # Load the saved model
    model = YOLO(model_path)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

                # Draw bounding boxes on the image with a thinner line
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Reduced thickness to 1

                # Put text with a smaller font size and increased thickness for better visibility
                font_scale = 0.4  # Reduced font scale
                thickness = 1      # Increased thickness for the text
                cv2.putText(img, f"{results[0].names[class_id]} {confidence:.2f}",
                            (x1, y1 - 5),  # Slightly adjusted y position
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            # Resize the image to a medium size (e.g., 800x600)
            resized_img = cv2.resize(img, (800, 600))

            # Save the resized annotated image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized_img)

    print("All images have been processed and saved to the output directory.")

# Example usage
model_path = 'C:/Users/vijay/OneDrive/Desktop/WDD/runs/detect/ir_defect_detection3/weights/best.pt'
image_dir = 'C:/Users/vijay/OneDrive/Desktop/WDD/datasets/ir_images/ir'
output_dir = 'C:/Users/vijay/OneDrive/Desktop/WDD/output/detected_defects'
detect_defects(model_path, image_dir, output_dir)
