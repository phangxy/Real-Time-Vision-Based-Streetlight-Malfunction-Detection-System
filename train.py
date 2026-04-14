import os
import torch
from ultralytics import YOLO

# --- SETUP ---
Current_directory = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\YOLOv8l"
Dataset_directory = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\Training Code"
DATASET_YAML = f"{Dataset_directory}/YOLOv8_datasets/DETECTION_Label-WITH-other-light-2/data.yaml"

def main():
    # Change directory (According to your dataset location)
    os.chdir(Current_directory)
    print("\n--- Starting Training ---")

    # Load model
    model = YOLO("yolov8l.pt")
    model.info()

    # Train
    model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=8,           
        device=0,
        workers=4,
        lr0=0.001,         # Start with a smaller learning rate (default is 0.01)
        warmup_epochs=5    # longer warmup to prevent early collapse
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
