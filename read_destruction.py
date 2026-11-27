"""Read the final destruction percentage from a recognized results screen."""

import argparse
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from recognize_hud import (
    BATTLE_REFERENCE,
    REFERENCE,
    RESULT_REFERENCE,
    normalized_image,
    recognize,
)

OCR_SCRIPT = Path(__file__).resolve().with_name("read_windows_ocr.ps1")
RESULT_NUMBER_BOX = (448, 233, 552, 264)
# Normalized 1000x800 boxes for the 3 large result stars (top tip of star 2 avoids the center % text).
RESULT_STAR_BOXES = (
    (345, 225, 415, 295),
    (470, 172, 530, 212),
    (585, 225, 655, 295),
)


def read_stars(image: Image.Image) -> int:
    """Count earned silver/white stars (0..3) on a normalized 1000x800 result screen (<0.2ms, zero OCR)."""
    arr = np.asarray(image.convert("RGB"))
    stars = 0
    for x0, y0, x1, y1 in RESULT_STAR_BOXES:
        patch = arr[y0:y1, x0:x1]
        mx = patch.max(axis=2)
        mn = patch.min(axis=2)
        silver_frac = float(((mx >= 175) & ((mx - mn) <= 55)).mean())
        if silver_frac >= 0.20:
            stars += 1
    return stars


def parse_percentage(text):
    normalized = unicodedata.normalize("NFKC", text)
    compact = "".join(normalized.split())
    match = re.fullmatch(r"([0-9]{1,3})%?", compact)
    if match and int(match[1]) <= 100:
        return int(match[1])
    # Handle garbled trailing '%' (e.g. '84 ' when WinRT OCR misreads the % glyph)
    nums = re.findall(r"[0-9]{1,3}", compact)
    if len(nums) == 1 and int(nums[0]) <= 100:
        return int(nums[0])
    return None


def read_result(image, work_dir):
    crop = image.crop(RESULT_NUMBER_BOX)
    pixels = list(crop.get_flattened_data() if hasattr(crop, "get_flattened_data") else crop.getdata())
    estimates = []
    for index, threshold in enumerate((135, 138, 125)):
        mask = Image.new("L", crop.size)
        mask.putdata([
            0 if g > threshold and g > r + 12 and g > b + 55 else 255
            for r, g, b in pixels
        ])
        if ImageOps.invert(mask).getbbox() is None:
            estimates.append(None)
            continue
        glyphs = mask.resize((crop.width * 6, crop.height * 6))
        with tempfile.TemporaryDirectory(dir=work_dir, prefix="destruction-ocr-") as temp:
            path = Path(temp) / f"mask-{index}.png"
            ImageOps.expand(glyphs, border=40, fill=255).save(path)
            output = subprocess.check_output(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 str(OCR_SCRIPT), str(path)],
                encoding="utf-8",
                timeout=15,
            )
        val = parse_percentage(output)
        estimates.append(val)
        if index == 1 and estimates[0] is not None and estimates[0] == estimates[1]:
            return estimates[0]
    valid = [v for v in estimates if v is not None]
    if not valid:
        return 100 if read_stars(image) == 3 else None
    for v in valid:
        if valid.count(v) >= 2:
            return v
    return valid[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--output", type=Path, help="Write the result to a new JSON file")
    args = parser.parse_args()
    image, size = normalized_image(args.screenshot)
    templates = [normalized_image(path)[0] for path in (REFERENCE, BATTLE_REFERENCE, RESULT_REFERENCE)]
    phase = recognize(image, size, *templates)["phase"]
    percentage = None
    status = "not_result"
    if phase == "result":
        try:
            percentage = read_result(image, OCR_SCRIPT.parent)
            status = "ok" if percentage is not None else "unreadable"
        except (OSError, subprocess.SubprocessError, UnicodeError):
            status = "ocr_error"
    result = {
        "screenshot": str(args.screenshot.resolve()),
        "phase": phase,
        "final_destruction": percentage,
        "status": status,
    }
    payload = json.dumps(result)
    if args.output:
        with args.output.open("x", encoding="utf-8") as record:
            record.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
