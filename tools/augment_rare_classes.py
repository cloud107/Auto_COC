"""Copy-Paste rare-class booster for CoC fixed-isometric YOLO datasets.

Boosts underrepresented classes (e.g., guardian_smasher, guardian_log_thrower, or rare
defending hero flags flag_*) up to a target instance count in images/train by pasting
feathered real crops onto non-overlapping battlefield positions and appending exact YOLO labels.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _load_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes


def _iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1e-6)


def _feathered_paste(dst: np.ndarray, crop: np.ndarray, x1: int, y1: int, feather_px: int = 3) -> None:
    h, w = crop.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    if feather_px > 0 and h > 2 * feather_px and w > 2 * feather_px:
        for k in range(feather_px):
            alpha = (k + 1) / (feather_px + 1)
            mask[k, :] = np.minimum(mask[k, :], alpha)
            mask[h - 1 - k, :] = np.minimum(mask[h - 1 - k, :], alpha)
            mask[:, k] = np.minimum(mask[:, k], alpha)
            mask[:, w - 1 - k] = np.minimum(mask[:, w - 1 - k], alpha)
    alpha3 = mask[:, :, None]
    roi = dst[y1:y1 + h, x1:x1 + w].astype(np.float32)
    dst[y1:y1 + h, x1:x1 + w] = np.clip(alpha3 * crop.astype(np.float32) + (1.0 - alpha3) * roi, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="YOLO dataset root containing images/train & labels/train")
    parser.add_argument("--class-ids", type=int, nargs="+", required=True, help="Class IDs to boost (e.g. 20 22 45 46)")
    parser.add_argument("--target", type=int, default=65, help="Target total box count per boosted class in train")
    parser.add_argument("--max-per-image", type=int, default=2, help="Max synthetic pastes per generated image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    img_dir = args.dataset / "images" / "train"
    lbl_dir = args.dataset / "labels" / "train"
    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"Expected images/train and labels/train under {args.dataset}")

    # Collect existing real images (ignore .json, classes.txt, and previously generated aug_cp_* files)
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    train_imgs = [
        p for p in sorted(img_dir.iterdir())
        if p.is_file() and p.suffix.lower() in img_exts and not p.stem.startswith("aug_cp_")
    ]
    crops_by_cls: dict[int, list[tuple[np.ndarray, float, float]]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)

    for img_path in train_imgs:
        boxes = _load_yolo_boxes(lbl_dir / f"{img_path.stem}.txt")
        needed = [b for b in boxes if b[0] in args.class_ids]
        if not needed:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for cid, cx, cy, bw, bh in needed:
            counts[cid] += 1
            x1 = max(0, int(round((cx - bw / 2) * iw)))
            y1 = max(0, int(round((cy - bh / 2) * ih)))
            x2 = min(iw, int(round((cx + bw / 2) * iw)))
            y2 = min(ih, int(round((cy + bh / 2) * ih)))
            if x2 - x1 >= 10 and y2 - y1 >= 10:
                crops_by_cls[cid].append((img[y1:y2, x1:x2].copy(), bw, bh))

    deficits = {cid: max(0, args.target - counts[cid]) for cid in args.class_ids}
    print("Current train counts:", {cid: counts[cid] for cid in args.class_ids})
    print("Deficits to generate:", deficits)
    if sum(deficits.values()) == 0:
        print("All requested classes already meet --target.")
        return

    gen_idx = 0
    while any(v > 0 for v in deficits.values()):
        base_path = rng.choice(train_imgs)
        base_img = cv2.imread(str(base_path))
        if base_img is None:
            continue
        ih, iw = base_img.shape[:2]
        boxes = _load_yolo_boxes(lbl_dir / f"{base_path.stem}.txt")
        pasted = 0
        cids_needed = [cid for cid, rem in deficits.items() if rem > 0 and crops_by_cls[cid]]
        if not cids_needed:
            break
        rng.shuffle(cids_needed)
        for cid in cids_needed[:args.max_per_image]:
            crop, bw, bh = rng.choice(crops_by_cls[cid])
            if rng.random() < 0.5:
                crop = cv2.flip(crop, 1)
            ch, cw = crop.shape[:2]
            # Sample inside the central playable diamond (x in 0.20..0.80, y in 0.16..0.72) avoiding overlaps
            placed = False
            for _ in range(40):
                nx = rng.uniform(0.20, 0.80)
                ny = rng.uniform(0.16, 0.72)
                if abs(nx - 0.50) / 0.36 + abs(ny - 0.44) / 0.34 > 0.95:
                    continue
                cand = (nx, ny, cw / iw, ch / ih)
                if any(_iou_xywh(cand, b[1:]) > 0.05 for b in boxes):
                    continue
                x1 = int(round(nx * iw - cw / 2))
                y1 = int(round(ny * ih - ch / 2))
                if 0 <= x1 and x1 + cw <= iw and 0 <= y1 and y1 + ch <= ih:
                    _feathered_paste(base_img, crop, x1, y1, feather_px=3)
                    boxes.append((cid, (x1 + cw / 2) / iw, (y1 + ch / 2) / ih, cw / iw, ch / ih))
                    deficits[cid] -= 1
                    pasted += 1
                    placed = True
                    break
            if not placed:
                continue
        if pasted > 0:
            gen_idx += 1
            stem = f"aug_cp_{gen_idx:04d}_{base_path.stem}"
            cv2.imwrite(str(img_dir / f"{stem}.png"), base_img)
            lbl_lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lbl_lines) + "\n", encoding="utf-8")

    print(f"Generated {gen_idx} Copy-Paste augmented train images. Remaining deficits: {deficits}")


if __name__ == "__main__":
    main()
