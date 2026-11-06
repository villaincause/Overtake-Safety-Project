# Overtake Safety Project

**Short description:** A computer-vision prototype that detects vehicles near a configurable safety line and signals an overtaking status through on-screen codes or optional Arduino LEDs.

## Overview

The project uses a YOLO model to detect vehicles in webcam or video footage. A vehicle is considered ahead when its bounding box crosses the configurable safety line. The highest applicable vehicle category determines the output:

| Code | Signal | Vehicle category |
|---|---|---|
| 1 | Green | No vehicle detected beyond the line |
| 2 | Small | Bicycle or bike |
| 3 | Blue | Car, CNG, rickshaw, van, or similar |
| 4 | Red | Bus, pickup, or truck |

This is an experimental prototype, not a certified vehicle safety system. Do not rely on it to make real-world driving decisions.

## Requirements

- Python 3.10 or newer
- A webcam for live detection, or a video file
- A trained model at `models/best.pt` for inference
- Optional: Arduino Nano and LEDs for hardware output

## Setup

Create and activate a virtual environment from the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install PyTorch for your computer. For CPU-only inference:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU, choose the appropriate PyTorch installation command from [pytorch.org](https://pytorch.org/get-started/locally/). Then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Model weights are not included in the repository. Place trained weights at `models/best.pt`, or train a model using the instructions below.

## Run inference

Start with the default webcam (camera 0):

```powershell
python realtime.py
```

Use a video file or select a camera:

```powershell
python realtime.py --source test_video.mp4
python realtime.py --source 1
```

Press **Q** in the video window to stop. Adjust the safety line with the slider, by dragging on the video, or with **W**/**S**. The selected line position is saved in `settings.json`.

To analyze a video and save an annotated output:

```powershell
python detect.py --video test_video.mp4
```

The output defaults to `outputs/overtake_safety_output.mp4`. Add `--show` to preview while processing. Run `python realtime.py --help` or `python detect.py --help` to see additional options.

## Train a model

Place a YOLO-format dataset ZIP at `data/Traffic Car Segmentation 2.v2i.yolov11.zip`, then run:

```powershell
python train.py
```

Training extracts the dataset under `data/dataset/`, saves run artifacts under `runs/`, and copies the best weights to `models/best.pt`. You can also customize training, for example:

```powershell
python train.py --epochs 100 --batch 8
```

## Optional Arduino LEDs

Upload [`arduino/OvertakeLEDs/OvertakeLEDs.ino`](arduino/OvertakeLEDs/OvertakeLEDs.ino) to the Arduino and connect it over USB. Close the Arduino IDE Serial Monitor, then run:

```powershell
python realtime.py --port auto
```

To specify a serial port manually, use `--port COM3` (replace `COM3` as needed). The sketch uses 9600 baud by default. Its LED outputs correspond to codes 1–4; all LEDs blink if the computer stops sending updates.

## Troubleshooting

- Check available cameras with `python realtime.py --list-cameras`.
- Check available serial ports with `python realtime.py --list-ports`.
- Run `python diagnose.py` to check the Python environment, libraries, camera, serial ports, and model.
- If inference is slow on CPU, try `python realtime.py --imgsz 320`.
