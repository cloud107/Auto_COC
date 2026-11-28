"""Train the 44-class pre-battle full-building yolo26m detector on Windows CUDA GPU.

Warm-starts from the proven 24-class defense checkpoint (cls_remap=True preserves backbone/neck
and classes 0..23 while learning the 20 new resource/army/utility/tesla classes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="D:/AI4COC/models/realistic-defense24-yolo26m-self-20261125.pt")
    parser.add_argument("--data", default="D:/AI4COC/datasets/prebattle-full-v1/data.yaml")
    parser.add_argument("--project", default="D:/AI4COC/runs")
    parser.add_argument("--name", default="prebattle-full44-yolo26m-20261129")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path("D:/AI4COC/yolo26m.pt")

    print(f"[TRAIN] Loading base weights: {model_path}")
    print(f"[TRAIN] Dataset YAML: {args.data}")
    model = YOLO(str(model_path))

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="0",
        workers=0,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        cls_remap=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        patience=args.patience,
        save=True,
        save_period=10,
        amp=True,
        cache=False,
        max_det=300,
        # Preserve fixed CoC isometric building geometry
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        scale=0.35,
        translate=0.10,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,
        erasing=0.20,
        plots=True,
        verbose=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[TRAIN] Finished! Best weights: {best_pt}")
    if best_pt.exists():
        print("[EVAL] Running test-split evaluation on best.pt...")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device="0", workers=0, max_det=300)
        print(f"[TEST] mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
