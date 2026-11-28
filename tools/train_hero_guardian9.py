"""Train the 9-class Hero + Guardian detector (pre-battle & live-battle shared).

Classes (9):
  0: hero_bk, 1: hero_aq, 2: hero_gw, 3: hero_rc, 4: hero_mp, 5: hero_dd,
  6: guardian_smasher, 7: guardian_longshot, 8: guardian_log_thrower

Uses yolo26s (fast enough for live-battle multi-FPS tracking) warm-started from
CoC yolo26s weights (or yolo26s.pt fallback), with horizontal flip enabled because
walking heroes face both left and right.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="D:/AI4COC/models/realistic-defense20-yolo26s-r1.pt")
    parser.add_argument("--data", default="D:/AI4COC/datasets/hero-guardian-v1/data.yaml")
    parser.add_argument("--project", default="D:/AI4COC/runs")
    parser.add_argument("--name", default="hero-guardian9-yolo26s-20261129")
    parser.add_argument("--export-model", default="D:/AI4COC/models/hero-guardian9-yolo26s-20261129.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.0006)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    model_path = Path(args.model)
    load_target = str(model_path) if model_path.exists() else "yolo26s.pt"
    print(f"[TRAIN] Loading base weights: {load_target}")
    print(f"[TRAIN] Dataset YAML: {args.data}")
    model = YOLO(load_target)

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
        max_det=50,
        # Walking heroes & guardians face both left and right -> fliplr=0.5 helps generalize!
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.3,
        close_mosaic=15,
        mixup=0.0,
        scale=0.35,
        translate=0.10,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,
        erasing=0.15,
        plots=True,
        verbose=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[TRAIN] Finished! Best weights: {best_pt}")
    if best_pt.exists():
        dst = Path(args.export_model)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dst)
        print(f"[EXPORT] Copied best weights -> {dst}")
        print("[EVAL] Running test-split evaluation on best.pt...")
        best_model = YOLO(str(best_pt))
        metrics = best_model.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device="0", workers=0, max_det=50)
        print(f"[TEST] mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
