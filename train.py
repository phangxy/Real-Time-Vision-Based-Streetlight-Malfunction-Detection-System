import os
import torch
from ultralytics import YOLO

# --- SETUP ---
HOME = r"C:\Users\Xin Yun\Desktop\From OLD\Universiti Malaya\FYP\Training Code"
DATASET_YAML = f"{HOME}/datasets/DETECTION_Label-WITH-other-light-2/data.yaml"

def main():
    # Change directory (according to your dataset location)
    os.chdir(HOME)

    print("\n--- Starting Training ---")

    # Load model
    model = YOLO("yolo11l.pt")
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
