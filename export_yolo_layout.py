"""Run a YOLO detector on a game viewport and export its pixel boxes."""

import argparse
import json
from pathlib import Path


def layout_from_result(result, screenshot, weights, smoke_test, class_map=None):
    height, width = result.orig_shape
    boxes = result.boxes
    if boxes is None:
        raise ValueError("Expected detection boxes from a YOLO Detect model")
    detections = [
        {
            "class_id": int(class_id),
            "class": result.names[int(class_id)],
            "bbox_xyxy": coords,
            "confidence": confidence,
        }
        for coords, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
        )
    ]
    if class_map is not None:
        for detection in detections:
            detection["building_class_id"] = class_map[detection["class"]]
    return {
        "source_image": screenshot.name,
        "coordinate_system": "viewport_pixels",
        "viewport": {"width": width, "height": height},
        "detector": "yolo",
        "weights": weights.name,
        "coverage": "smoke_test_only" if smoke_test else "unverified",
        "detections": detections,
    }


def intersection_over_union(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def predict_layout(model, screenshot, weights, confidence, device, sizes, smoke_test=False,
                   class_map=None, viewport_width=None):
    sizes = list(dict.fromkeys(sizes))
    source = str(screenshot)
    native_size = None
    if viewport_width is not None:
        from PIL import Image

        with Image.open(screenshot) as frame:
            native_size = frame.size
            viewport_height = round(frame.height * viewport_width / frame.width)
            source = frame.convert("RGB").resize((viewport_width, viewport_height), Image.Resampling.LANCZOS)
    merged = []
    for size in sizes:
        result, = model.predict(source, imgsz=size, conf=confidence, device=device, verbose=False)
        layout = layout_from_result(result, screenshot, weights, smoke_test, class_map)
        merged.extend(layout["detections"])
    layout["inference_sizes"] = sizes
    if native_size is not None:
        scale_x = native_size[0] / layout["viewport"]["width"]
        scale_y = native_size[1] / layout["viewport"]["height"]
        for detection in merged:
            detection["bbox_xyxy"] = [
                value * (scale_x if index % 2 == 0 else scale_y)
                for index, value in enumerate(detection["bbox_xyxy"])
            ]
        layout["viewport"] = {"width": native_size[0], "height": native_size[1]}
        layout["normalized_viewport_width"] = viewport_width
    if len(sizes) == 1:
        return layout
    kept = []
    for candidate in sorted(merged, key=lambda detection: detection["confidence"], reverse=True):
        if any(candidate["class"] == detection["class"] and intersection_over_union(
                candidate["bbox_xyxy"], detection["bbox_xyxy"]) >= .50 for detection in kept):
            continue
        kept.append(candidate)
    layout["detections"] = kept
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--extra-imgsz", type=int, action="append", default=[],
                        help="Additional image size for multiscale inference")
    parser.add_argument("--viewport-width", type=int,
                        help="Resize to the training viewport width while preserving source pixel coordinates")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--class-map", type=Path,
                        help="Optional map from model class names to building schema IDs")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Mark generic model predictions as smoke test data")
    args = parser.parse_args()
    if not args.screenshot.is_file() or not args.weights.is_file():
        parser.error("Both the screenshot and the weights must be local files")
    if min(args.imgsz, *args.extra_imgsz, args.viewport_width if args.viewport_width is not None else 1) <= 0 or not 0 <= args.confidence <= 1:
        parser.error("Expected a positive image size and confidence between 0 and 1")
    from ultralytics import YOLO

    model = YOLO(str(args.weights), task="detect")
    class_map = None
    if args.class_map:
        class_map = json.loads(args.class_map.read_text(encoding="utf-8"))
        missing = sorted(set(model.names.values()) - set(class_map))
        if missing:
            parser.error(f"Model classes absent from class map: {', '.join(missing)}")
    layout = predict_layout(model, args.screenshot, args.weights, args.confidence, args.device,
                            [args.imgsz, *args.extra_imgsz], args.smoke_test, class_map, args.viewport_width)
    args.output.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
