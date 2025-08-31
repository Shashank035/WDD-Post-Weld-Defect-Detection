import os
import cv2
import numpy as np

def load_images(input_folder):
    images = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)  # Read image in color
            img_resized = cv2.resize(img, (256, 256))  # Resize to match model input size
            img_normalized = img_resized / 255.0  # Normalize
            images.append(img_normalized)
            print(f"Loaded image: {filename}")
    
    return images

def convert_to_ir(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply colormap to simulate IR
            ir_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            
            # Save the IR image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, ir_img)
            print(f"Converted {filename} to IR and saved at {output_path}")

# Load masked images and convert to IR-style images
input_folder = r"C:\Users\vijay\OneDrive\Desktop\WDD\datasets\masked_images"
output_folder = r"C:\Users\vijay\OneDrive\Desktop\WDD\datasets\ir_images\ir"

# Load images (optional, not used in conversion)
masked_images = load_images(input_folder)

# Convert masked images to IR-style images
convert_to_ir(input_folder, output_folder)
