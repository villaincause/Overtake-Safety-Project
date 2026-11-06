"""
Train a YOLO11 segmentation model locally and save the best weights.

Folder layout (relative to this file):
    data/Traffic Car Segmentation 2.v2i.yolov11.zip   <- put your dataset zip here
    models/best.pt                                    <- created after training
    runs/overtake*/                                   <- Ultralytics training output

Usage:
    python train.py
    python train.py --epochs 100 --batch 8
    python train.py --resume runs/overtake/weights/last.pt
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_ZIP = ROOT / "data" / "Traffic Car Segmentation 2.v2i.yolov11.zip"
DATASET_DIR = ROOT / "data" / "dataset"
MODELS_DIR = ROOT / "models"


def pick_device():
    """CUDA GPU if available, then Apple Silicon, otherwise CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_yaml(folder: Path):
    for name in ("data.yaml", "data.yml"):
        matches = sorted(folder.rglob(name))
        if matches:
            return matches[0]
    return None


def prepare_dataset(zip_path: Path, dataset_dir: Path) -> Path:
    """Unzip the dataset once and write a data_local.yaml with correct absolute paths.

    The original data.yaml from the zip is left untouched.
    """
    if find_yaml(dataset_dir) is None:
        if not zip_path.exists():
            sys.exit(
                f"Dataset zip not found: {zip_path}\n"
                "Put it there, or pass --zip /path/to/dataset.zip"
            )
        print(f"Extracting {zip_path.name} ...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dataset_dir)

    src_yaml = find_yaml(dataset_dir)
    if src_yaml is None:
        sys.exit(f"Could not find data.yaml inside {dataset_dir}")

    base = src_yaml.parent.resolve()
    with open(src_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Roboflow exports often use "../train/images", which only resolves correctly
    # from some locations. Try the entry as written, then with leading "../" removed.
    for split in ("train", "val", "test"):
        entry = config.get(split)
        if not entry:
            continue
        stripped = str(entry)
        while stripped.startswith(("../", "..\\")):
            stripped = stripped[3:]
        resolved = next(
            ((base / c).resolve() for c in (str(entry), stripped) if (base / c).exists()),
            None,
        )
        if resolved is None:
            if split == "test":
                config.pop("test")  # test split is optional
                continue
            sys.exit(f"Could not locate '{split}' images (data.yaml says: {entry}) under {base}")
        config[split] = resolved.as_posix()

    config["path"] = base.as_posix()
    out_yaml = base / "data_local.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Dataset ready. YAML: {out_yaml}")
    print(config)
    return out_yaml


def main():
    p = argparse.ArgumentParser(description="Train YOLO11-seg and save best weights")
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="dataset zip (Roboflow YOLO export)")
    p.add_argument("--model", default="yolo11n-seg.pt",
                   help="'yolo11n-seg.pt' for polygon masks, 'yolo11n.pt' for plain boxes")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="lower this if you run out of GPU memory (-1 = auto)")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers (use 0 if you hit errors on Windows)")
    p.add_argument("--patience", type=int, default=20, help="stop early if no improvement for N epochs")
    p.add_argument("--device", default=None, help="0, 'cpu', 'mps' ... (default: auto-detect)")
    p.add_argument("--resume", type=Path, default=None, help="path to a last.pt to resume an interrupted run")
    args = p.parse_args()

    from ultralytics import YOLO  # imported here so --help works instantly

    if args.resume:
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        data_yaml = prepare_dataset(args.zip, DATASET_DIR)
        device = args.device if args.device is not None else pick_device()
        print(f"Training on device: {device}")
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            patience=args.patience,
            project=str(ROOT / "runs"),
            name="overtake",  # Ultralytics auto-numbers: overtake, overtake2, ...
        )

    # Ultralytics tracks the best epoch (by fitness) and writes it as best.pt
    best = Path(model.trainer.best)
    MODELS_DIR.mkdir(exist_ok=True)
    final = MODELS_DIR / "best.pt"
    shutil.copy2(best, final)
    print(f"\nBest epoch weights : {best}")
    print(f"Copied to          : {final}")


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
