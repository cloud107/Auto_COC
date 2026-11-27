"""Read the final destruction percentage from a recognized results screen."""

import argparse
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

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


def parse_percentage(text):
    normalized = unicodedata.normalize("NFKC", text)
    compact = "".join(normalized.split())
    match = re.fullmatch(r"([0-9]{1,3})%", compact)
    if match and int(match[1]) <= 100:
        return int(match[1])
    return None


def read_result(image, work_dir):
    crop = image.crop(RESULT_NUMBER_BOX)
    estimates = []
    for index, threshold in enumerate((135, 138)):
        mask = Image.new("L", crop.size)
        mask.putdata([
            0 if g > threshold and g > r + 12 and g > b + 55 else 255
            for r, g, b in (crop.get_flattened_data() if hasattr(crop, "get_flattened_data") else crop.getdata())
        ])
        bounds = ImageOps.invert(mask).getbbox()
        if bounds is None:
            return None
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
        estimates.append(parse_percentage(output))
    return estimates[0] if estimates[0] == estimates[1] else None


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
