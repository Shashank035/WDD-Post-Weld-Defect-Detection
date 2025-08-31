from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model on your custom dataset
    model.train(
        data='C:/Users/vijay/OneDrive/Desktop/WDD/weld_defect_detection.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='ir_defect_detection'
    )

if __name__ == '__main__':
    main()
