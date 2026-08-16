# train.py  (this will run on Kaggle)
from ultralytics import YOLO
import os
import shutil

# ---------- CONFIG ----------
DATA_PATH = "/kaggle/input/license-plate-recognition/License Plate Recognition"  # Kaggle dataset mount
OUT_DIR = "/kaggle/working/output_model"
os.makedirs(OUT_DIR, exist_ok=True)

# Ensure ultralytics installed on kernel
# Kaggle kernels allow pip install in notebook/script; include it if needed
try:
    import ultralytics
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

# Load pretrained YOLOv8m from PyPI (Ultralytics will download)
model = YOLO("yolov8s.pt")   # pretrained weights for faster convergence

data_yaml = os.path.join(DATA_PATH, "data.yaml")  # your dataset should include data.yaml

# Training (adjust epochs/batch for Kaggle T4)
model.train(
    data=data_yaml,
    epochs=70,
    imgsz=640,
    batch=16,
    device="0,1",
    workers=2,
    amp=True,
    cache=True,
    name="yolov8_trained_model",
    project="."   # default runs/ folder will be created here
)

# After training, export the best model to ONNX (or .pt)
trained_dir = "runs/train/yolov8_trained_model"
best_pt = None
for root, dirs, files in os.walk(trained_dir):
    for f in files:
        if f.endswith(".pt") and "best" in f:
            best_pt = os.path.join(root, f)
            break
    if best_pt:
        break

if best_pt:
    # Export to ONNX
    export_path = os.path.join(OUT_DIR, "best.onnx")
    model.export(format="onnx", weights=best_pt, imgsz=640, device="0")
    # If ultralytics exported to runs/export, copy it
    # Try copying standard exported file location
    possible = os.path.join(trained_dir, "weights", "best.onnx")
    if os.path.exists(possible):
        shutil.copy(possible, export_path)
else:
    # fallback: export the final weights the model object has
    export_path = os.path.join(OUT_DIR, "final.onnx")
    model.export(format="onnx", imgsz=640, device="0")
