"""Collect unique pre-deployment screenshots from the private base generator."""

import argparse
import io
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from recognize_hud import BATTLE_REFERENCE, REFERENCE, RESULT_REFERENCE, normalized_image, recognize


def fingerprint(image):
    return image.crop((80, 65, 920, 650)).convert("L").resize((48, 34), Image.Resampling.BOX)


def read_frame(adb, templates):
    png = subprocess.check_output([*adb, "exec-out", "screencap", "-p"], timeout=10)
    image, size = normalized_image(io.BytesIO(png))
    if size != (1800, 1440):
        raise RuntimeError(f"Expected a calibrated 1800×1440 screenshot, got {size}")
    return png, image, recognize(image, size, *templates)["phase"]


def wait_for_phase(adb, templates, expected):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        png, image, phase = read_frame(adb, templates)
        if phase == expected:
            return png, image
        time.sleep(.4)
    raise RuntimeError(f"Expected {expected}, last recognized phase: {phase}")


def taps(adb, instructions):
    subprocess.run([*adb, "shell", instructions], check=True, timeout=18)


def leave_battle(adb, templates):
    taps(adb, "input tap 143 1238; sleep 0.55; input tap 1117 882")
    wait_for_phase(adb, templates, "result")


def write_manifest(output):
    manifest = output / "review.json"
    prior_rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    existing = {row["image"] for row in prior_rows}
    rows = list(prior_rows)
    for path in sorted((output / "images").glob("realistic-*.png")):
        relative = f"images/{path.name}"
        if relative in existing:
            continue
        number = int(path.stem.rsplit("-", 1)[1])
        split = {0: "test", 5: "val"}.get(number % 10, "train")
        rows.append(dict(layout_id=path.stem, image=relative, split=split,
                         reviewed_classes=[], boxes=[]))
    pending = output / "review.tmp"
    pending.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    pending.replace(manifest)


def adb_zoom_out(adb, dev="/dev/input/event4", repeats=2, steps=6):
    """Zoom all the way out purely via Android multi-touch pinch over ADB (no Windows foreground/mouse needed!)."""
    cmds = []
    # /dev/input/event4 ("Xiaomi Touchscreen") uses portrait MT coordinates X:0..1440, Y:0..1800 centered at (720, 900).
    x1_0, y1_0, x2_0, y2_0 = 240, 380, 1200, 1420
    x1_1, y1_1, x2_1, y2_1 = 660, 840, 780, 960
    for r in range(repeats):
        tid0, tid1 = 100 + r * 2, 101 + r * 2
        cmds += [
            f"sendevent {dev} 3 47 0",
            f"sendevent {dev} 3 57 {tid0}",
            f"sendevent {dev} 3 53 {x1_0}",
            f"sendevent {dev} 3 54 {y1_0}",
            f"sendevent {dev} 1 325 1",
            f"sendevent {dev} 0 0 0",
            f"sendevent {dev} 3 47 1",
            f"sendevent {dev} 3 57 {tid1}",
            f"sendevent {dev} 3 53 {x2_0}",
            f"sendevent {dev} 3 54 {y2_0}",
            f"sendevent {dev} 0 0 0",
        ]
        for s in range(1, steps + 1):
            a = s / steps
            cx1, cy1 = round(x1_0 + a * (x1_1 - x1_0)), round(y1_0 + a * (y1_1 - y1_0))
            cx2, cy2 = round(x2_0 + a * (x2_1 - x2_0)), round(y2_0 + a * (y2_1 - y2_0))
            cmds += [
                f"sendevent {dev} 3 47 0",
                f"sendevent {dev} 3 53 {cx1}",
                f"sendevent {dev} 3 54 {cy1}",
                f"sendevent {dev} 3 47 1",
                f"sendevent {dev} 3 53 {cx2}",
                f"sendevent {dev} 3 54 {cy2}",
                f"sendevent {dev} 0 0 0",
                "sleep 0.02",
            ]
        cmds += [
            f"sendevent {dev} 3 47 1",
            f"sendevent {dev} 3 57 -1",
            f"sendevent {dev} 0 0 0",
            f"sendevent {dev} 3 47 0",
            f"sendevent {dev} 3 57 -1",
            f"sendevent {dev} 1 325 0",
            f"sendevent {dev} 0 0 0",
            "sleep 0.12",
        ]
    script = "\n".join(cmds) + "\n"
    subprocess.run([*adb, "shell"], input=script.encode("ascii"), check=True, timeout=15)


def zoom_out(result, adb=None):
    if adb is not None:
        adb_zoom_out(adb)
        return
    result.unlink(missing_ok=True)
    subprocess.run(["schtasks", "/Run", "/TN", "AI4COC-MuMuZoom"], check=True, timeout=6,
                   stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        if result.exists():
            message = result.read_text(encoding="utf-8-sig").strip()
            if message == "ok":
                return
            if message:
                raise RuntimeError(f"MuMu zoom failed: {message}")
        time.sleep(.2)
    raise RuntimeError("Timed out waiting for MuMu zoom")


def collect(adb, output, count, zoom_result):
    templates = [normalized_image(path)[0] for path in (REFERENCE, BATTLE_REFERENCE, RESULT_REFERENCE)]
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    saved = sorted(images.glob("realistic-*.png"))
    signatures = [fingerprint(normalized_image(path)[0]) for path in saved]
    write_manifest(output)
    if len(saved) >= count:
        return
    next_number = max((int(path.stem.rsplit("-", 1)[1]) for path in saved), default=0) + 1
    _, _, phase = read_frame(adb, templates)
    if phase in ("pre_deployment", "battle"):
        leave_battle(adb, templates)
        phase = "result"

    duplicate_streak = 0
    while len(signatures) < count:
        if phase == "result":
            taps(adb, "input tap 902 1193; sleep 2.7")
        taps(adb, "input tap 615 1365; sleep 1.7; input tap 497 710; sleep 1.0; "
                  "input tap 1417 880; sleep 0.35; input tap 481 769; sleep 0.35; "
                  "input tap 1344 1224")
        png, image = wait_for_phase(adb, templates, "pre_deployment")
        if zoom_result is not None:
            zoom_out(zoom_result)
            time.sleep(.4)
            png, image = wait_for_phase(adb, templates, "pre_deployment")
        sample = fingerprint(image)
        if any(ImageStat.Stat(ImageChops.difference(sample, old)).mean[0] < 3.0 for old in signatures):
            duplicate_streak += 1
            print(f"Skipped near-duplicate layout ({duplicate_streak} consecutive)", flush=True)
        else:
            target = images / f"realistic-{next_number:03d}.png"
            with target.open("xb") as file:
                file.write(png)
            signatures.append(sample)
            next_number += 1
            duplicate_streak = 0
            write_manifest(output)
            print(f"Captured {len(signatures)}/{count}: {target.name}", flush=True)
        leave_battle(adb, templates)
        phase = "result"
        if duplicate_streak >= 5:
            raise RuntimeError("Generator returned five consecutive near-duplicate layouts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--adb", default="D:/MuMuPlayer/nx_main/adb.exe")
    parser.add_argument("--serial", default="127.0.0.1:16384")
    parser.add_argument("--zoom-result", type=Path, help="Result file from the interactive MuMu zoom task")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    subprocess.run([args.adb, "connect", args.serial], check=True, timeout=10)
    collect([args.adb, "-s", args.serial], args.output, args.count, args.zoom_result)


if __name__ == "__main__":
    main()
