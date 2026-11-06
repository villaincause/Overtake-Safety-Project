"""
Run this on any PC to check whether the project can run there, and how fast.
    python diagnose.py
If something fails, copy the whole output and send it.
"""
import platform
import sys
import time
from pathlib import Path

import numpy as np

import realtime as rt

ROOT = Path(__file__).resolve().parent


def section(t):
    print(f"\n=== {t} ===")


section("System")
print("Python   :", sys.version.split()[0], platform.platform())
try:
    import os
    print("CPU cores:", os.cpu_count(), "|", platform.processor())
except Exception:
    pass

section("Libraries")
for mod in ("torch", "ultralytics", "cv2", "numpy"):
    try:
        m = __import__(mod)
        print(f"{mod:12}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:12}: MISSING / BROKEN ({e})")
try:
    import torch
    print("CUDA GPU : ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none -> will run on CPU")
except Exception:
    pass

section("OpenCV window support")
try:
    rt.cv2.namedWindow("t")
    rt.cv2.destroyAllWindows()
    print("OK (window can be shown)")
except Exception as e:
    print("NO WINDOW SUPPORT:", str(e).strip().splitlines()[0])
    print("Fix: pip uninstall opencv-python-headless -y && pip install opencv-python")

section("Cameras")
rt.list_cameras()

section("Serial ports (Arduino)")
try:
    rt.list_serial_ports()
except Exception as e:
    print("pyserial problem:", e, "| fix: pip install pyserial")

section("Model + speed (blank 640x480 frame, ~average of 10 runs)")
model_path = ROOT / "models" / "best.pt"
if not model_path.exists():
    print(f"models/best.pt not found at {model_path}")
else:
    try:
        det = rt.Detector(model_path, "auto", None, 0.4)
        print("device:", det.device)
        frame = np.zeros((480, 640, 3), np.uint8)
        det(frame)  # warm-up
        for size in (320, 416, 640):
            det.imgsz = size
            t0 = time.time()
            for _ in range(10):
                det(frame)
            ms = (time.time() - t0) / 10 * 1000
            print(f"imgsz {size}: {ms:6.1f} ms/frame  (~{1000 / ms:.1f} FPS)")
        print("Class names:", list(det.names.values()))
    except Exception as e:
        print("MODEL FAILED TO LOAD/RUN:", repr(e))
