import os
import cv2

# Specify the directory path
directory_path = r'C:\Users\vijay\OneDrive\Desktop\WDD\datasets\ir_images\ir'

# Initialize a counter for renaming
counter = 1

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image
        img = cv2.imread(os.path.join(directory_path, filename))
        
        # Construct the new filename
        new_filename = f"image{counter}.jpg"
        
        # Save the image as .jpg
        cv2.imwrite(os.path.join(directory_path, new_filename), img)
        
        # Remove the original file
        os.remove(os.path.join(directory_path, filename))
        
        print(f"Renamed and converted {filename} to {new_filename}")
        
        # Increment the counter
        counter += 1
