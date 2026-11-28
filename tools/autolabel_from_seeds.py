"""Semi-automatic X-AnyLabeling pre-labeler using multi-seed, multi-scale BGR template matching.

Harvests already-labeled crops for target classes from seed images in a workspace (e.g. 5 seed
images of hero_hall & helper_hut), searches all remaining unlabeled images using 3-channel BGR
TM_CCOEFF_NORMED across slight zoom scales, excludes regions already covered by existing boxes,
renders a visual QA contact sheet, and optionally appends verified predictions into X-AnyLabeling
.json files without touching any existing annotations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _shape_bbox(shape: dict) -> tuple[float, float, float, float] | None:
    pts = shape.get("points", [])
    if len(pts) < 2:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _overlaps_existing(cand: tuple[float, float, float, float], existing: list[tuple[float, float, float, float]]) -> bool:
    cx, cy = (cand[0] + cand[2]) * 0.5, (cand[1] + cand[3]) * 0.5
    for eb in existing:
        if _iou(cand, eb) >= 0.22:
            return True
        # Reject if candidate center sits strictly inside an existing box (or vice-versa)
        if eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3]:
            return True
        ecx, ecy = (eb[0] + eb[2]) * 0.5, (eb[1] + eb[3]) * 0.5
        if cand[0] <= ecx <= cand[2] and cand[1] <= ecy <= cand[3]:
            return True
    return False


def _hs_hist(bgr: np.ndarray) -> np.ndarray:
    """Compute normalized 2D Hue-Saturation histogram for color-accurate verification (rejects wrong-color matches)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _save_contact_sheet(items: list[tuple[float, str, np.ndarray]], out_path: Path, tile_size: int = 128, cols: int = 6):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * (tile_size + 26), cols * tile_size, 3), 24, dtype=np.uint8)
    for idx, (score, title, crop) in enumerate(items):
        r, c = divmod(idx, cols)
        y0, x0 = r * (tile_size + 26), c * tile_size
        resized = cv2.resize(crop, (tile_size - 4, tile_size - 4), interpolation=cv2.INTER_AREA)
        canvas[y0 + 2:y0 + tile_size - 2, x0 + 2:x0 + tile_size - 2] = resized
        col = (80, 255, 100) if score >= 0.62 else ((60, 220, 255) if score >= 0.50 else (70, 90, 255))
        cv2.putText(canvas, f"{score:.2f} {title[:12]}", (x0 + 3, y0 + tile_size + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Workspace root (e.g. datasets/prebattle-full-v1)")
    parser.add_argument("--classes", nargs="+", required=True, help="Class names to auto-label (e.g. hero_hall helper_hut)")
    parser.add_argument("--max-per-image", type=int, default=1, help="Max instances per image per class (1 for unique buildings)")
    parser.add_argument("--min-score", type=float, default=0.52, help="Min BGR TM_CCOEFF_NORMED score to accept")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.93, 0.965, 1.0, 1.035, 1.07])
    parser.add_argument("--downscale", type=float, default=0.5, help="Search downscale factor (0.5 = 900x720)")
    parser.add_argument("--trim", type=float, default=0.08, help="Outer border trim fraction when matching template")
    parser.add_argument("--max-templates", type=int, default=8, help="Max seed templates used per class for speed")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Directory to save QA contact sheets")
    parser.add_argument("--apply", action="store_true", help="Write accepted predictions into X-AnyLabeling .json files")
    args = parser.parse_args()

    json_paths = sorted(args.dataset.glob("images/*/*.json"))
    if not json_paths:
        raise SystemExit(f"No .json files found under {args.dataset / 'images'}")

    # Step 1: Harvest seed templates from all images that already contain any requested class
    templates: dict[str, list[dict]] = defaultdict(list)
    existing_data: dict[Path, dict] = {}
    for jp in json_paths:
        data = json.loads(jp.read_text(encoding="utf-8"))
        existing_data[jp] = data
        img_path = jp.with_name(data["imagePath"])
        if not img_path.exists():
            continue
        seed_shapes = [sh for sh in data.get("shapes", []) if sh.get("label") in args.classes and sh.get("description") != "auto_template"]
        if not seed_shapes:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for sh in seed_shapes:
            box = _shape_bbox(sh)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            if bw < 16 or bh < 16:
                continue
            # Trim outer margin slightly so varying neighboring grass/walls don't dilute NCC score
            mx, my = bw * args.trim, bh * args.trim
            tx1, ty1 = max(0, int(round(x1 + mx))), max(0, int(round(y1 + my)))
            tx2, ty2 = min(iw, int(round(x2 - mx))), min(ih, int(round(y2 - my)))
            crop = img[ty1:ty2, tx1:tx2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            templates[sh["label"]].append({
                "crop": crop,
                "hist": _hs_hist(crop),
                "full_w": bw,
                "full_h": bh,
                "trim_x": tx1 - x1,
                "trim_y": ty1 - y1,
                "src": jp.stem,
            })

    for cls in args.classes:
        if len(templates[cls]) > args.max_templates:
            step = len(templates[cls]) / args.max_templates
            templates[cls] = [templates[cls][int(i * step)] for i in range(args.max_templates)]
        print(f"[SEEDS] {cls}: {len(templates[cls])} templates from {[t['src'] for t in templates[cls]]}")
        if not templates[cls]:
            raise SystemExit(f"No seed annotations found for class '{cls}'!")

    # Pre-build downscaled multi-scale templates
    ds = args.downscale
    scaled_tmpls: dict[str, list[tuple[np.ndarray, float, dict]]] = defaultdict(list)
    for cls in args.classes:
        for t in templates[cls]:
            for sc in args.scales:
                tw = max(8, int(round(t["crop"].shape[1] * ds * sc)))
                th = max(8, int(round(t["crop"].shape[0] * ds * sc)))
                resized = cv2.resize(t["crop"], (tw, th), interpolation=cv2.INTER_AREA)
                scaled_tmpls[cls].append((resized, sc, t))

    # Step 2: Search all images missing each class
    accepted_by_cls: dict[str, list[tuple[float, str, np.ndarray]]] = defaultdict(list)
    rejected_by_cls: dict[str, list[tuple[float, str, np.ndarray]]] = defaultdict(list)
    new_shapes_by_jp: dict[Path, list[dict]] = defaultdict(list)

    for jp in json_paths:
        data = existing_data[jp]
        shapes = data.get("shapes", [])
        img_path = jp.with_name(data["imagePath"])
        if not img_path.exists():
            continue
        cls_needed = [
            cls for cls in args.classes
            if sum(1 for sh in shapes if sh.get("label") == cls) < args.max_per_image
        ]
        if not cls_needed:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        img_small = cv2.resize(img, (int(round(iw * ds)), int(round(ih * ds))), interpolation=cv2.INTER_AREA)
        sh_s, sw_s = img_small.shape[:2]
        existing_boxes = [b for sh in shapes if (b := _shape_bbox(sh)) is not None]

        for cls in cls_needed:
            already_cnt = sum(1 for sh in shapes if sh.get("label") == cls)
            need = args.max_per_image - already_cnt
            cands: list[tuple[float, tuple[float, float, float, float]]] = []
            for tmpl_s, sc, meta in scaled_tmpls[cls]:
                th, tw = tmpl_s.shape[:2]
                if th >= sh_s or tw >= sw_s:
                    continue
                res = cv2.matchTemplate(img_small, tmpl_s, cv2.TM_CCOEFF_NORMED)
                # Strictly mask out surrounding forest trees, ocean water, and HUD (keep only playable village diamond!)
                ry_g, rx_g = np.mgrid[:res.shape[0], :res.shape[1]]
                nx = (rx_g + tw * 0.5) / float(sw_s)
                ny = (ry_g + th * 0.5) / float(sh_s)
                in_village = (
                    (np.abs(nx - 0.50) / 0.43 + np.abs(ny - 0.46) / 0.40 <= 1.02)
                    & (ny >= 0.06)
                    & (ny <= 0.84)
                )
                res[~in_village] = -1.0
                # Extract top-8 peaks per template/scale with local suppression
                work = res.copy()
                for _ in range(8):
                    _, maxv, _, (mx, my) = cv2.minMaxLoc(work)
                    if maxv < max(0.35, args.min_score - 0.12):
                        break
                    # Suppress neighborhood around this peak
                    sy0, sy1 = max(0, my - th // 2), min(work.shape[0], my + th // 2 + 1)
                    sx0, sx1 = max(0, mx - tw // 2), min(work.shape[1], mx + tw // 2 + 1)
                    work[sy0:sy1, sx0:sx1] = -1.0
                    # Reconstruct full-resolution bounding box
                    x1 = (mx / ds) - meta["trim_x"] * sc
                    y1 = (my / ds) - meta["trim_y"] * sc
                    x2 = x1 + meta["full_w"] * sc
                    y2 = y1 + meta["full_h"] * sc
                    box = (max(0.0, x1), max(0.0, y1), min(float(iw), x2), min(float(ih), y2))
                    if _overlaps_existing(box, existing_boxes):
                        continue
                    # Color-histogram check on inner candidate crop rejects wrong-color shapes (purple cannon / green tree / wall)
                    bx1, by1, bx2, by2 = map(int, map(round, box))
                    mx_c = int(round((bx2 - bx1) * args.trim))
                    my_c = int(round((by2 - by1) * args.trim))
                    inner = img[by1 + my_c:max(by1 + my_c + 4, by2 - my_c), bx1 + mx_c:max(bx1 + mx_c + 4, bx2 - mx_c)]
                    if inner.size == 0:
                        continue
                    chist = _hs_hist(inner)
                    h_sim = max(float(cv2.compareHist(chist, t["hist"], cv2.HISTCMP_CORREL)) for t in templates[cls])
                    if h_sim < 0.45:
                        continue
                    # Combined shape (BGR NCC) + color-histogram score
                    combined = 0.72 * float(maxv) + 0.28 * max(0.0, h_sim)
                    cands.append((combined, box))

            cands.sort(key=lambda x: x[0], reverse=True)
            picked: list[tuple[float, tuple[float, float, float, float]]] = []
            for score, box in cands:
                if any(_iou(box, pb) >= 0.30 for _, pb in picked):
                    continue
                picked.append((score, box))
                if len(picked) >= need:
                    break

            for score, box in picked:
                x1, y1, x2, y2 = box
                pad = 12
                cx1, cy1 = max(0, int(round(x1)) - pad), max(0, int(round(y1)) - pad)
                cx2, cy2 = min(iw, int(round(x2)) + pad), min(ih, int(round(y2)) + pad)
                vis_crop = img[cy1:cy2, cx1:cx2].copy()
                # Draw tight predicted box in green on preview crop
                cv2.rectangle(
                    vis_crop,
                    (int(round(x1)) - cx1, int(round(y1)) - cy1),
                    (int(round(x2)) - cx1, int(round(y2)) - cy1),
                    (0, 255, 0), 2,
                )
                if score >= args.min_score:
                    accepted_by_cls[cls].append((score, jp.stem[-10:], vis_crop))
                    existing_boxes.append(box)
                    new_shapes_by_jp[jp].append({
                        "label": cls,
                        "score": round(score, 3),
                        "points": [
                            [round(x1, 2), round(y1, 2)],
                            [round(x2, 2), round(y1, 2)],
                            [round(x2, 2), round(y2, 2)],
                            [round(x1, 2), round(y2, 2)],
                        ],
                        "group_id": None,
                        "description": "auto_template",
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": {},
                        "attributes": {},
                    })
                else:
                    rejected_by_cls[cls].append((score, jp.stem[-10:], vis_crop))

    # Step 3: Print QA stats and save contact sheets (worst accepted + best rejected!)
    for cls in args.classes:
        acc = sorted(accepted_by_cls[cls], key=lambda x: x[0])
        rej = sorted(rejected_by_cls[cls], key=lambda x: x[0], reverse=True)
        scores = [s for s, _, _ in acc]
        s_min = f"{min(scores):.3f}" if scores else "N/A"
        s_med = f"{float(np.median(scores)):.3f}" if scores else "N/A"
        s_max = f"{max(scores):.3f}" if scores else "N/A"
        print(f"[RESULT] {cls:18s}: accepted={len(acc):2d} (min={s_min}, med={s_med}, max={s_max}) | rejected={len(rej):2d}")
        if args.preview_dir is not None:
            # Show the 18 LOWEST-scoring accepted predictions (hardest cases!) + top 6 rejected
            sample_acc = acc[:18]
            _save_contact_sheet(sample_acc, args.preview_dir / f"{cls}_lowest_accepted.jpg")
            if rej:
                _save_contact_sheet(rej[:12], args.preview_dir / f"{cls}_top_rejected.jpg")

    if args.apply:
        written = 0
        for jp, added in new_shapes_by_jp.items():
            data = existing_data[jp]
            data["shapes"].extend(added)
            jp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written += len(added)
        print(f"[APPLIED] Wrote {written} new boxes across {len(new_shapes_by_jp)} X-AnyLabeling .json files.")
    else:
        print("[DRY-RUN] No .json files modified. Re-run with --apply after inspecting preview contact sheets.")


if __name__ == "__main__":
    main()
