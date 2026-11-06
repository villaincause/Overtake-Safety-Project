"""
Real-time overtaking-safety check: webcam (or video file) in, one number out.

Output codes (printed to stdout whenever the code changes):
    1 = GREEN   no vehicle ahead, safe to pass
    2 = SMALL   small vehicle ahead   (bikes, bycycle)   [temporary spare LED for now]
    3 = BLUE    medium vehicle ahead  (cng, auto_rickshaw, cars, mini_van, van, rickshaw)
    4 = RED     large vehicle ahead   (bus, pickup, trucks)

A vehicle counts as "ahead" when its bottom edge is below the safety line.
If several vehicles are past the line, the LARGEST category wins (4 > 3 > 2).

Moving the safety line while running (it is saved to settings.json on exit):
    - drag the "Safety line %" slider, or
    - left-click / drag anywhere in the video, or
    - press  w = up   s = down   (1% per press),   q = quit

Runs on any PC: uses an NVIDIA GPU if there is one, otherwise the CPU (smaller
input size, always processes the NEWEST camera frame so it never lags behind).

Usage:
    python realtime.py                       # webcam 0
    python realtime.py --list-cameras        # which camera numbers work?
    python realtime.py --source 1            # second camera
    python realtime.py --source test_video.mp4
    python realtime.py --imgsz 320           # faster on slow PCs
    python realtime.py --device cpu          # force CPU
    python realtime.py --no-display          # headless, Ctrl+C to stop
    python realtime.py --virtual-leds        # draw the 4 LEDs on screen (test without an Arduino)

Arduino (LED display) over USB serial:
    python realtime.py --list-ports          # which COM ports exist?
    python realtime.py --port auto           # find the Arduino Nano automatically
    python realtime.py --port COM3           # or name the port yourself
The code (1-4) is sent as text like "3\\n", repeated every 0.5 s as a heartbeat. "0" = all off.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "settings.json"
WINDOW = "Overtake Safety"
TRACKBAR = "Safety line %"

# Code 2 = small-vehicle LED. When its final color is chosen, change its name and color here.
CODE_NAMES = {1: "GREEN", 2: "SMALL", 3: "BLUE", 4: "RED"}
CODE_COLORS_BGR = {2: (255, 0, 255), 3: (255, 0, 0), 4: (0, 0, 255)}  # BGR order: magenta, blue, red

# Class names come from the dataset's data.yaml (matching is case-insensitive).
CLASS_TO_CODE = {
    # small -> 2 (small-vehicle LED)
    "bikes": 2, "bycycle": 2, "bicycle": 2,
    # medium -> 3 (blue)
    "cng": 3, "auto_rickshaw": 3, "cars": 3, "mini_van": 3, "van": 3, "rickshaw": 3,
    # large -> 4 (red)
    "bus": 4, "pickup": 4, "trucks": 4,
}


def log(*a):
    """Status messages go to stderr so stdout carries only the codes."""
    print(*a, file=sys.stderr, flush=True)


def normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def decide_code(detections, line_y):
    """detections: iterable of (class_name, y2). Returns 1-4."""
    code = 1
    for name, y2 in detections:
        if y2 > line_y:
            code = max(code, CLASS_TO_CODE.get(normalize(name), 1))
    return code


class Stabilizer:
    """Only switch the output after the new code was seen N frames in a row,
    so the LEDs don't flicker when a detection drops out for a single frame."""

    def __init__(self, frames: int):
        self.frames = max(1, frames)
        self.current = None
        self.candidate = None
        self.count = 0

    def update(self, raw: int):
        if raw == self.current:
            self.candidate, self.count = None, 0
        elif raw == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = raw, 1
        if self.candidate is not None and self.count >= self.frames:
            self.current, self.candidate, self.count = self.candidate, None, 0
        return self.current


def draw_virtual_leds(frame, code):
    """On-screen copy of the 4 Arduino LEDs, for testing without hardware (--virtual-leds)."""
    h, w = frame.shape[:2]
    colors = {1: (0, 200, 0), **CODE_COLORS_BGR}
    step, r = 52, 18
    x0, y0 = w - 4 * step - 6, 14
    cv2.rectangle(frame, (x0 - 8, y0 - 8), (w - 6, y0 + 2 * r + 30), (30, 30, 30), -1)
    for i in range(1, 5):
        cx, cy = x0 + (i - 1) * step + r, y0 + r
        lit = code == i
        cv2.circle(frame, (cx, cy), r, colors[i] if lit else tuple(c // 5 for c in colors[i]), -1)
        cv2.circle(frame, (cx, cy), r, (255, 255, 255) if lit else (90, 90, 90), 2)
        cv2.putText(frame, str(i), (cx - 6, cy + r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def emit(code: int):
    """THE HOOK: called once each time the stable code changes.
    For now it just prints the number. Later, send it to the Arduino here
    (e.g. serial.write(f"{code}\\n".encode()))."""
    print(code, flush=True)


def load_line_pct(default: int) -> int:
    try:
        return int(json.loads(SETTINGS_PATH.read_text())["safety_line_pct"])
    except Exception:
        return default


def save_line_pct(pct: int):
    SETTINGS_PATH.write_text(json.dumps({"safety_line_pct": pct}, indent=2))


# ----------------------------------------------------------------------------
# Arduino link (USB serial)
# ----------------------------------------------------------------------------
KNOWN_USB_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}  # Arduino, Arduino.org, CH340, FTDI, CP210x


def serial_ports():
    from serial.tools import list_ports
    return list(list_ports.comports())


def list_serial_ports():
    ports = serial_ports()
    if not ports:
        log("  no serial ports found (is the Arduino plugged in? CH340 driver installed?)")
    for p in ports:
        vid = f"{p.vid:04X}" if p.vid is not None else "----"
        log(f"  {p.device}: {p.description} (USB vendor id {vid})")
    return ports


def find_arduino_port():
    """Best guess for an Arduino Nano: genuine boards, CH340 / FTDI / CP210x clones."""
    for p in serial_ports():
        text = f"{p.description} {p.manufacturer}".lower()
        if p.vid in KNOWN_USB_VIDS or any(k in text for k in ("arduino", "ch340", "ch341", "ftdi", "usb-serial", "usb serial")):
            return p.device
    return None


class ArduinoLink:
    """Sends the code to the Arduino. Sends immediately on change, and repeats the current
    code every RESEND_S seconds. The Arduino treats silence as 'PC stopped' and blinks all LEDs."""

    RESEND_S = 0.5
    RETRY_S = 3.0

    def __init__(self, port: str, baud: int):
        self.port_arg = port
        self.baud = baud
        self.ser = None
        self.port = None
        self.last_code = None
        self.last_send = 0.0
        self.next_retry = 0.0

    def connect(self) -> bool:
        import serial

        port = find_arduino_port() if self.port_arg == "auto" else self.port_arg
        if not port:
            log("Could not find an Arduino. Plug it in and run:  python realtime.py --list-ports")
            return False
        kwargs = dict(baudrate=self.baud, timeout=0)
        if "://" not in port:  # real COM ports only: network ports (rfc2217://) don't support write_timeout
            kwargs["write_timeout"] = 1
        try:
            self.ser = serial.serial_for_url(port, **kwargs)
        except (serial.SerialException, OSError, ValueError, NotImplementedError) as e:
            log(f"Could not open {port}: {e}\n  (Is the Arduino IDE Serial Monitor still open? It must be closed.)")
            self.ser = None
            return False
        time.sleep(2.0)  # opening the port restarts the Nano; give it time to boot
        self.port = port
        self.last_code = None
        log(f"Arduino connected on {port}")
        return True

    def send(self, code: int):
        import serial

        if self.ser is None:
            if time.time() < self.next_retry:
                return
            if not self.connect():
                self.next_retry = time.time() + self.RETRY_S
                return
        try:
            self.ser.write(f"{code}\n".encode())
            self.last_code, self.last_send = code, time.time()
        except (serial.SerialException, OSError) as e:
            log(f"Arduino link lost ({e}). Will retry every {self.RETRY_S:.0f} s.")
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            self.next_retry = time.time() + self.RETRY_S

    def heartbeat(self, code: int):
        if code != self.last_code or time.time() - self.last_send >= self.RESEND_S:
            self.send(code)

    def close(self):
        if self.ser is not None:
            try:
                self.ser.write(b"0\n")  # all LEDs off
                self.ser.flush()
                self.ser.close()
            except Exception:
                pass
            self.ser = None


# ----------------------------------------------------------------------------
# Model / device handling
# ----------------------------------------------------------------------------
def resolve_device(arg: str):
    """'auto' -> NVIDIA GPU 0 if PyTorch can see one, otherwise CPU."""
    if arg != "auto":
        return int(arg) if arg.isdigit() else arg
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def exported_imgsz(model_path: Path):
    """Exported models (e.g. OpenVINO folder) have a fixed input size stored in metadata.yaml."""
    try:
        import yaml
        meta = yaml.safe_load((model_path / "metadata.yaml").read_text())
        size = meta.get("imgsz")
        return int(size[0] if isinstance(size, (list, tuple)) else size)
    except Exception:
        return None


class Detector:
    """Wraps the YOLO model. Picks GPU/CPU automatically and falls back to CPU
    if the GPU path fails at runtime."""

    def __init__(self, model_path: Path, device: str, imgsz, conf: float):
        from ultralytics import YOLO

        self._YOLO = YOLO
        self.path = model_path
        self.conf = conf
        self.is_pt = model_path.suffix == ".pt"
        self.device = resolve_device(device) if self.is_pt else None
        self.auto_imgsz = imgsz is None
        if self.is_pt:
            self.imgsz = imgsz or (640 if self.device != "cpu" else 416)
        else:  # exported model: input size is fixed at export time
            self.imgsz = exported_imgsz(model_path) or imgsz or 640
        self.model = YOLO(str(model_path))

    @property
    def names(self):
        return self.model.names

    def __call__(self, frame):
        kw = dict(conf=self.conf, imgsz=self.imgsz, verbose=False, max_det=30)
        if self.device is not None:
            kw["device"] = self.device
        try:
            return self.model(frame, **kw)[0]
        except Exception as e:
            if self.device in (None, "cpu"):
                raise
            log(f"GPU inference failed ({e}). Falling back to CPU.")
            self.device = "cpu"
            if self.auto_imgsz:
                self.imgsz = 416
            self.model = self._YOLO(str(self.path))  # reload cleanly on CPU
            kw.update(device="cpu", imgsz=self.imgsz)
            return self.model(frame, **kw)[0]


# ----------------------------------------------------------------------------
# Camera / video input
# ----------------------------------------------------------------------------
def open_camera(index: int, width: int, height: int):
    """Open a webcam, trying several backends on Windows (DirectShow often works
    where the default backend fails or starts slowly)."""
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def list_cameras(width=640, height=480, max_index=5):
    found = []
    for i in range(max_index):
        cap = open_camera(i, width, height)
        if cap is not None:
            found.append(i)
            log(f"  camera {i}: OK ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
            cap.release()
    if not found:
        log("  no working cameras found (is it plugged in / used by another app?)")
    return found


class LatestFrameGrabber:
    """Reads the camera in a background thread and always hands over the NEWEST frame.
    Without this, a slow PC falls behind and the video/codes lag further and further."""

    MAX_FAILS = 60

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.fresh = False
        self.alive = True
        self.stopped = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fails = 0
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails >= self.MAX_FAILS:
                    self.alive = False
                    return
                time.sleep(0.02)
                continue
            fails = 0
            with self.lock:
                self.frame, self.fresh = frame, True

    def read(self, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.fresh:
                    self.fresh = False
                    return True, self.frame
            if not self.alive:
                break
            time.sleep(0.002)
        return False, None

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()


def open_source(source: str, width: int, height: int):
    """Returns an object with .read() and .release(), or None."""
    if source.isdigit():
        index = int(source)
        cap = open_camera(index, width, height)
        if cap is None:
            log(f"Camera {index} did not open. Scanning for other cameras...")
            for i in list_cameras(width, height):
                if i != index:
                    log(f"Using camera {i} instead (use --source {i} to pick it explicitly).")
                    cap = open_camera(i, width, height)
                    break
        return LatestFrameGrabber(cap) if cap is not None else None
    cap = cv2.VideoCapture(source)  # video file: read every frame in order
    return cap if cap.isOpened() else None


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file path")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt",
                   help="a .pt file, or an exported OpenVINO folder")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--line", type=int, default=None,
                   help="safety line start position, %% of frame height from the top "
                        "(default: last saved value, else 65)")
    p.add_argument("--stable-frames", type=int, default=3,
                   help="frames a new code must persist before it is output (1 = no smoothing)")
    p.add_argument("--device", default="auto", help="auto, cpu, or a GPU number like 0")
    p.add_argument("--imgsz", type=int, default=None,
                   help="model input size, multiple of 32 (default: 640 on GPU, 416 on CPU; "
                        "try 320 on a very slow PC)")
    p.add_argument("--width", type=int, default=640, help="camera capture width")
    p.add_argument("--height", type=int, default=480, help="camera capture height")
    p.add_argument("--list-cameras", action="store_true", help="show which camera numbers work, then exit")
    p.add_argument("--port", default=None,
                   help="Arduino serial port: 'auto' or e.g. COM3 (default: no Arduino, numbers only print)")
    p.add_argument("--baud", type=int, default=9600, help="serial speed (must match the Arduino sketch)")
    p.add_argument("--list-ports", action="store_true", help="show serial ports, then exit")
    p.add_argument("--virtual-leds", action="store_true",
                   help="draw the 4 LEDs in the video window (test without an Arduino)")
    p.add_argument("--no-display", action="store_true", help="run without a window")
    args = p.parse_args()

    if args.list_cameras:
        log("Checking cameras...")
        list_cameras(args.width, args.height)
        return

    if args.list_ports:
        log("Serial ports:")
        list_serial_ports()
        return

    try:  # don't let Ultralytics try to reach the internet at startup (demo venue may be offline)
        from ultralytics import settings
        settings.update({"sync": False})
    except Exception:
        pass

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model} (run train.py first)")

    log("Loading model...")
    detector = Detector(args.model, args.device, args.imgsz, args.conf)
    log(f"Model ready. device={detector.device if detector.device is not None else 'n/a (exported)'}, "
        f"imgsz={detector.imgsz}")
    unmapped = sorted(n for n in detector.names.values() if normalize(n) not in CLASS_TO_CODE)
    if unmapped:
        log(f"WARNING: these model classes have no LED code and will be ignored: {unmapped}")

    reader = open_source(args.source, args.width, args.height)
    if reader is None:
        sys.exit(f"Could not open source: {args.source}")

    link = None
    if args.port:
        link = ArduinoLink(args.port, args.baud)
        if not link.connect():
            reader.release()
            sys.exit("Arduino connection failed (see message above). Not starting without the LEDs.")

    display = not args.no_display
    line_pct = args.line if args.line is not None else load_line_pct(65)
    line_pct = max(0, min(100, line_pct))
    ui = {"h": 0}

    def on_mouse(event, x, y, flags, _):
        dragging = event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        if (event == cv2.EVENT_LBUTTONDOWN or dragging) and ui["h"]:
            cv2.setTrackbarPos(TRACKBAR, WINDOW, max(0, min(100, round(100 * y / ui["h"]))))

    if display:
        try:
            cv2.namedWindow(WINDOW)
            cv2.createTrackbar(TRACKBAR, WINDOW, line_pct, 100, lambda v: None)
            cv2.setMouseCallback(WINDOW, on_mouse)
        except cv2.error as e:
            log("No display window available (OpenCV GUI missing?). Running headless; codes still print.\n"
                "  Fix: pip uninstall opencv-python-headless -y && pip install opencv-python\n"
                f"  ({str(e).strip().splitlines()[0]})")
            display = False

    stab = Stabilizer(args.stable_frames)
    last_emitted = None
    last_t = time.time()
    fps = 0.0
    log("Running. Press q in the window (or Ctrl+C) to stop.")

    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                log("No more frames (camera disconnected or video ended).")
                break
            h, w = frame.shape[:2]
            ui["h"] = h
            if display:
                line_pct = cv2.getTrackbarPos(TRACKBAR, WINDOW)
            line_y = int(h * line_pct / 100)

            result = detector(frame)
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = detector.names[int(box.cls[0])]
                    detections.append((name, y2))
                    if display:
                        code_here = CLASS_TO_CODE.get(normalize(name), 1)
                        past = y2 > line_y and code_here > 1
                        color = CODE_COLORS_BGR[code_here] if past else (200, 200, 200)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{name} {float(box.conf[0]):.2f}", (x1, max(12, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            code = stab.update(decide_code(detections, line_y))
            if code is not None and code != last_emitted:
                emit(code)
                last_emitted = code
            if link is not None and code is not None:
                link.heartbeat(code)

            if display:
                cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 3)
                cv2.putText(frame, "SAFETY LINE (click / drag / w,s)", (20, max(20, line_y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6)
                last_t = now
                txt = f"CODE: {code if code else '--'}" + (f" ({CODE_NAMES[code]})" if code else "")
                cv2.putText(frame, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"{fps:.0f} FPS   line {line_pct}%", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if args.virtual_leds:
                    draw_virtual_leds(frame, code)
                if link is not None:
                    status = f"ARDUINO: {link.port}" if link.ser is not None else "ARDUINO: DISCONNECTED"
                    cv2.putText(frame, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("w"):
                    cv2.setTrackbarPos(TRACKBAR, WINDOW, max(0, line_pct - 1))
                elif key == ord("s"):
                    cv2.setTrackbarPos(TRACKBAR, WINDOW, min(100, line_pct + 1))
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        if link is not None:
            link.close()
        if display:
            cv2.destroyAllWindows()
            save_line_pct(line_pct)
            log(f"Safety line saved at {line_pct}%")


if __name__ == "__main__":
    main()
