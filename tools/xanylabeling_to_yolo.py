"""Convert X-AnyLabeling .json files in images/<split> back to YOLO labels/<split>/*.txt + data.yaml."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Workspace root (e.g. datasets/prebattle-full-v1)")
    args = parser.parse_args()

    classes_file = args.dataset / "classes.txt"
    if not classes_file.exists():
        raise SystemExit(f"classes.txt not found in {args.dataset}")
    class_names = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    name_to_id = {name: i for i, name in enumerate(class_names)}

    counts = Counter()
    unknown = Counter()
    converted = 0
    for split in ("train", "val", "test"):
        img_dir = args.dataset / "images" / split
        lbl_dir = args.dataset / "labels" / split
        if not img_dir.exists():
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for jp in sorted(img_dir.glob("*.json")):
            data = json.loads(jp.read_text(encoding="utf-8"))
            iw = float(data["imageWidth"])
            ih = float(data["imageHeight"])
            lines = []
            for sh in data.get("shapes", []):
                label = str(sh.get("label", "")).strip()
                if label not in name_to_id:
                    unknown[label] += 1
                    continue
                pts = sh.get("points", [])
                if len(pts) < 2:
                    continue
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                x1, x2 = max(0.0, min(xs)), min(iw, max(xs))
                y1, y2 = max(0.0, min(ys)), min(ih, max(ys))
                bw, bh = (x2 - x1) / iw, (y2 - y1) / ih
                if bw <= 1e-4 or bh <= 1e-4:
                    continue
                cx, cy = ((x1 + x2) * 0.5) / iw, ((y1 + y2) * 0.5) / ih
                cid = name_to_id[label]
                counts[cid] += 1
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_dir / f"{jp.stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
            converted += 1

    yaml_lines = [
        f"path: {args.dataset.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(class_names)],
        "",
    ]
    (args.dataset / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    print(f"Converted {converted} X-AnyLabeling .json files -> YOLO labels ({sum(counts.values())} boxes).")
    if unknown:
        print("WARNING - Unknown labels not in classes.txt (skipped):", dict(unknown))
    for cid, name in enumerate(class_names):
        if counts[cid] > 0:
            print(f"  {cid:2d}: {name:25s} -> {counts[cid]:4d}")


if __name__ == "__main__":
    main()
