"""
Run the overtaking-safety check on a video using the trained model.

Usage:
    python detect.py --video test_video.mp4
    python detect.py --video test_video.mp4 --show          # live preview, press q to quit
    python detect.py --video test_video.mp4 --line 0.70 --conf 0.5
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True, help="input video")
    p.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "overtake_safety_output.mp4")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--line", type=float, default=0.65,
                   help="safety line position as a fraction of frame height (from the top)")
    p.add_argument("--show", action="store_true", help="show a live preview window")
    args = p.parse_args()

    from ultralytics import YOLO

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model} (run train.py first)")
    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_name(args.output.stem + "_raw.mp4")
    out = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Any vehicle whose bottom edge extends below this line is too close (UNSAFE)
    SAFETY_LINE_Y = int(height * args.line)

    print("Processing video frames...")
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]
        is_unsafe = False

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if y2 > SAFETY_LINE_Y:
                    is_unsafe = True
                    box_color = (0, 0, 255)      # red: too close
                else:
                    box_color = (255, 255, 0)    # cyan: safe distance

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Safety threshold line
        cv2.line(frame, (0, SAFETY_LINE_Y), (width, SAFETY_LINE_Y), (0, 255, 255), 3)
        cv2.putText(frame, "SAFETY ZONE THRESHOLD", (20, SAFETY_LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # HUD signal light panel
        cv2.rectangle(frame, (width - 240, 20), (width - 20, 130), (30, 30, 30), -1)
        cv2.rectangle(frame, (width - 240, 20), (width - 20, 130), (255, 255, 255), 1)

        if is_unsafe:
            cv2.circle(frame, (width - 130, 55), 22, (0, 255, 255), -1)
            cv2.circle(frame, (width - 130, 55), 22, (255, 255, 255), 2)
            cv2.putText(frame, "WAIT (UNSAFE)", (width - 220, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.circle(frame, (width - 130, 55), 22, (0, 255, 0), -1)
            cv2.circle(frame, (width - 130, 55), 22, (255, 255, 255), 2)
            cv2.putText(frame, "SAFE TO OVERTAKE", (width - 230, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        out.write(frame)
        frame_no += 1
        if frame_no % 50 == 0:
            print(f"  {frame_no}/{total if total > 0 else '?'} frames")

        if args.show:
            cv2.imshow("Overtake Safety (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    # mp4v files often won't play in VS Code / browsers; re-encode to H.264 if ffmpeg exists
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
             str(args.output), "-loglevel", "quiet"],
            check=True,
        )
        raw_path.unlink()
    else:
        raw_path.replace(args.output)
        print("(ffmpeg not found: output is mp4v and may not play in some players)")

    print(f"Video processing completed! Saved output to: {args.output}")


if __name__ == "__main__":
    main()
