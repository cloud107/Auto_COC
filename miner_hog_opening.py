"""Run the fixed TH19 Miner/Hog opening using the current reference army."""
import argparse
import json
import math
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from battle_video_stream import BattleVideoStream
from capture_realistic_layouts import leave_battle, read_frame, taps, wait_for_phase, zoom_out
from dragon_ice_block import VIEWPORT, BattleCamera, DefenseFire, DragonTracker, IceBlockPolicy, battlefield_frame
from export_yolo_layout import predict_layout
from grid_projection import projection_for_viewport
from healing_spell import HealingMap, HealingPolicy, HealingWaveTracker
from outer_lawn_deployment import outer_lawn_cells
from recognize_hud import (
    BATTLE_REFERENCE, CARD_CENTERS, COUNT_BITS, NORMALIZED_SIZE, REFERENCE, RESULT_REFERENCE,
    normalized_image, patch_error, quantity_mask, quantity_templates, read_quantity, recognize,
)

ROOT = Path(__file__).resolve().parent
HEAL_SLOT = 11

CARDS = {
    "miner_squad": 1, "hog_squad": 2, "ice_golem": 3, "druid": 5, "siege_barracks": 6,
    "dragon_duke": 7, "archer_queen": 8, "grand_warden": 9, "royal_champion": 10, "ice_block": 14,
}


def edge_opening(cells, bounds, target):
    """Pick a usable side and three separated lanes, preferring the Town Hall side."""
    left, right, top, bottom = bounds
    edges = (
        [cell for cell in cells if cell[0]+.5 < left],
        [cell for cell in cells if cell[0]+.5 > right],
        [cell for cell in cells if cell[1]+.5 < top],
        [cell for cell in cells if cell[1]+.5 > bottom],
    )
    plans = []
    for side, edge in enumerate(edges):
        axis = 1 if side < 2 else 0
        by_position = sorted(edge, key=lambda cell: cell[axis])
        if len(by_position) < 12 or by_position[-1][axis] - by_position[0][axis] < 15:
            continue
        center_index = min(range(5, len(by_position)-5),
                           key=lambda index: math.dist(by_position[index], target))
        center = by_position[center_index]
        flanks = [
            min(by_position, key=lambda cell: abs(cell[axis] - (center[axis] + offset)))
            for offset in (-7, 7)
        ]
        if min(abs(cell[axis]-center[axis]) for cell in flanks) < 5:
            continue
        plans.append((math.dist(center, target), side, flanks, center, by_position, axis))
    if not plans:
        raise RuntimeError("No exterior edge offers two flanks and a main entry")
    _, side, flanks, center, edge, axis = min(plans)
    main = [min(edge, key=lambda cell: abs(cell[axis] - (center[axis] + offset)))
            for offset in (-2, -1, 1, 2)]
    return side, flanks, center, main


def card_change(before, after, slot):
    x = CARD_CENTERS[slot-1]
    region = (x-20, 746, x+20, 792)
    return round(patch_error(before.crop(region), after.crop(region)), 1)


def collect_predeployment(adb, templates, zoom, commands_open):
    _, _, phase = read_frame(adb, templates)
    if phase in ("pre_deployment", "battle") and not commands_open:
        leave_battle(adb, templates)
        phase = "result"
    if phase == "result":
        taps(adb, "input tap 902 1193; sleep 3")
    elif phase != "unknown":
        raise RuntimeError("Expected the village or result, or use --ready for a prepared battle")
    if not commands_open:
        taps(adb, "input tap 615 1365; sleep 4.2")
    taps(adb, "input tap 902 710; sleep 1.5; input tap 1344 685")
    wait_for_phase(adb, templates, "pre_deployment")
    zoom_out(zoom, adb=adb)
    time.sleep(.4)
    return wait_for_phase(adb, templates, "pre_deployment")



def read_battle_frame(adb, templates, viewport):
    # On this emulator the 16-byte header precedes uncompressed RGBA pixels.
    started = time.monotonic()
    data = subprocess.check_output([*adb, "exec-out", "screencap"], timeout=5)
    if len(data) != 16 + viewport[0]*viewport[1]*4 or struct.unpack_from("<III", data) != (*viewport, 1):
        raise RuntimeError("Expected a MuMu 1800×1440 raw RGBA screenshot")
    source = Image.frombytes("RGBA", viewport, data[16:]).convert("RGB")
    image = source.resize(NORMALIZED_SIZE, Image.Resampling.BILINEAR)
    frame = battlefield_frame(source)
    phase = recognize(image, viewport, *templates)["phase"]
    return image, frame, phase, round((time.monotonic()-started)*1000, 1)


def cast_dragon_ice_blocks(adb, args, templates, viewport, slot_center, camera, fire, seed,
                           first_frame, stage, persist):
    tracker = DragonTracker(seed)
    policy = IceBlockPolicy(args.ice_block_cooldown)
    quantities = quantity_templates(*templates[:2])
    spell_center = CARD_CENTERS[CARDS["ice_block"]-1]
    with Image.open(ROOT / "captures" / "ice-block-selected-one.png") as reference:
        template = Image.new("RGB", NORMALIZED_SIZE)
        template.paste(reference, (spell_center+5, 732))
    selected_one = quantity_mask(template, spell_center)

    def remaining_spells(image):
        count = read_quantity(image, spell_center, quantities)
        if count is not None:
            return count
        for dy in (-1, 0, 1):
            if COUNT_BITS(quantity_mask(image, spell_center, dy=dy) ^ selected_one) <= 72:
                return 1
        return None

    samples, casts = [], []
    stage["ice_block"] = {"samples": samples, "casts": casts}
    started = time.monotonic()
    image, frame, phase, capture_ms = first_frame
    while phase == "battle" and time.monotonic()-started < args.dragon_window and policy.attempts < 2:
        now = time.monotonic()
        transform, gray = camera.align(frame)
        if transform is None:
            tracker.consecutive = 0
        hero = tracker.observe(frame, transform, now) if transform is not None else None
        position = hero.position.tolist() if hero else None
        eligible, possible = fire.estimate(position, transform, gray, now) if hero else ([], [])
        ready, score = policy.evaluate(now, hero.health_width if hero else None, eligible)
        sample = {"t": round(now-started, 2), "capture_ms": capture_ms, "aligned": transform is not None,
                  "position": position, "health_width": hero.health_width if hero else None,
                  "air_fire_candidates": eligible, "possible_air_fire": possible, "score": score}
        samples.append(sample)
        if ready:
            remaining = remaining_spells(image)
            sample["spell_quantity"] = remaining
            if remaining == 2 - policy.attempts and 0 <= position[0] < VIEWPORT[0] and 30 <= position[1] < 810:
                point = [round(v*dst/src) for v, dst, src in zip(position, viewport, VIEWPORT)]
                action = {"unit": "ice_block", "slot": CARDS["ice_block"], "points": [point]}
                event = {"number": policy.attempts + 1, "status": "dispatching",
                         "point": point, "quantity_before": remaining, "sample": len(samples)-1}
                casts.append(event)
                stage["commands"].append(action)
                persist()
                card_x, card_y = slot_center
                taps(adb, f"input tap {card_x} {card_y}; sleep 0.15; input tap {point[0]} {point[1]}")
                policy.record_attempt(time.monotonic())
                event["status"] = "tap_sent"
                persist()
                image.save(args.output / f"ice-block-{event['number']}-before.png")
        image, frame, phase, capture_ms = read_battle_frame(adb, templates, viewport)
        # The spell card can animate for several frames after the tap.
        if casts and "quantity_after" not in casts[-1] and phase == "battle":
            remaining = remaining_spells(image)
            if remaining == casts[-1]["quantity_before"]-1:
                casts[-1]["quantity_after"] = remaining
                casts[-1]["status"] = "confirmed_by_quantity"
                image.save(args.output / f"ice-block-{casts[-1]['number']}-after.png")
                persist()
    if casts and casts[-1]["quantity_before"] == 1 and phase == "battle":
        # A depleted card may show no digit. Save the settled card for verification.
        end = time.monotonic() + .9
        while phase == "battle" and time.monotonic() < end:
            image, _, phase, _ = read_battle_frame(adb, templates, viewport)
        image.save(args.output / "ice-block-2-after.png")
    stage["ice_block"]["exit"] = "two_attempts" if policy.attempts == 2 else (
        "window_elapsed" if phase == "battle" else phase
    )
    persist()


class BattleSample(NamedTuple):
    timestamp: float
    image: Image.Image
    frame: np.ndarray
    phase: str
    capture_ms: float
    source: str


def healing_frames(adb, templates, viewport, report):
    with ExitStack() as capture:
        try:
            stream = capture.enter_context(BattleVideoStream(adb))
        except OSError as exc:
            report["stream_error"] = str(exc)
            stream = None
        earliest = time.monotonic()
        while True:
            if stream is not None:
                requested = time.monotonic()
                try:
                    timestamp, frame = stream.read(earliest)
                except OSError as exc:
                    report["stream_error"] = str(exc)
                    capture.close()
                    stream = None
                else:
                    earliest = timestamp + 1/12
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                        NORMALIZED_SIZE, Image.Resampling.BILINEAR
                    )
                    phase = recognize(image, viewport, *templates)["phase"]
                    capture_ms = round((time.monotonic()-requested)*1000, 1)
                    yield BattleSample(timestamp, image, frame, phase, capture_ms, "video")
                    continue
            image, frame, phase, capture_ms = read_battle_frame(adb, templates, viewport)
            yield BattleSample(time.monotonic(), image, frame, phase, capture_ms, "screenshot")


def stage_card_changes(stage, initial_image, after, viewport, templates, persist):
    hud = recognize(after, viewport, *templates)
    remaining = {card["slot"]: card["remaining"] for card in hud["cards"]}
    stage["status"] = "sent"
    stage["card_changes"] = {
        cmd["unit"]: {"appearance_delta": card_change(initial_image, after, cmd["slot"]),
                      "remaining": remaining.get(cmd["slot"])}
        for cmd in stage["commands"]
    }
    persist()
    print(f"{stage['name']}: {stage['card_changes']}", flush=True)


def deploy_main_wave_with_healing(adb, args, templates, viewport, card, camera, terrain,
                                  layout, stage, initial_image, instructions, persist):
    report = {"casts": [], "samples": [], "during_deployment": []}
    stage["healing"] = report

    def dispatch():
        taps(adb, instructions)
        return time.monotonic()

    with closing(healing_frames(adb, templates, viewport, report)) as frames:
        before = next(frames)
        if before.phase != "battle":
            raise RuntimeError(f"Cannot deploy the main wave from {before.phase}")
        started = time.monotonic()
        stage["dispatch_started_monotonic_s"] = started
        report["camera_ready_offset_s"] = round(before.timestamp-started, 3)
        pre_bgr = cv2.cvtColor(battlefield_frame(initial_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        tracker = HealingWaveTracker(started, stage["commands"], layout, pre_bgr=pre_bgr)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(dispatch)
            while not pending.done():
                sample = next(frames)
                if sample.timestamp < started or sample.phase != "battle":
                    continue
                transform, _ = camera.align(sample.frame)
                groups = tracker.observe(sample.frame, sample.timestamp, transform=transform)
                report["during_deployment"].append({
                    "t": round(sample.timestamp-started, 3), "source": sample.source,
                    "groups": [{"id": g["id"], "center": g["center"], "source": g["source"],
                                "visible_bars": len(g.get("bars", []))} for g in groups],
                })
            finished = pending.result()
        stage["dispatch_finished_monotonic_s"] = finished
        after = None
        while time.monotonic()-finished < 5:
            sample = next(frames)
            if sample.phase == "result":
                break
            if sample.timestamp < finished or sample.phase != "battle":
                continue
            transform, _ = camera.align(sample.frame)
            tracker.observe(sample.frame, sample.timestamp, transform=transform)
            after = sample.image
            if all(card_change(initial_image, after, CARDS[unit]) >= 30 for unit in ("miner_squad", "hog_squad")):
                break
        if after is None:
            raise RuntimeError("Could not capture the main wave after dispatch")
        after.save(args.output / "main_wave.png")
        stage_card_changes(stage, initial_image, after, viewport, templates, persist)
        if all(stage["card_changes"][unit]["appearance_delta"] >= 30 for unit in ("miner_squad", "hog_squad")):
            cast_main_heals(adb, args, templates, viewport, card, camera, terrain, tracker, frames, report, persist)
        else:
            report["exit"] = "deployment_unverified"
            persist()


def cast_main_heals(adb, args, templates, viewport, card, camera, terrain, tracker, frames, report, persist):
    policy = HealingPolicy(terrain)
    attempts = report["casts"]
    start = time.monotonic()
    stock = card["remaining"]
    card_position = CARD_CENTERS[HEAL_SLOT-1]
    with Image.open(ROOT / "captures" / "heal-selected-one.png") as reference:
        template = Image.new("RGB", NORMALIZED_SIZE)
        template.paste(reference, (card_position+5, 732))
    selected_one = quantity_mask(template, card_position)
    unknown_since = None
    for sample in frames:
        timestamp, image, frame, phase, capture_ms, source = sample
        if len(attempts) >= stock or timestamp-start >= 160:
            break
        if phase != "battle":
            if unknown_since is None:
                unknown_since = timestamp
            if phase == "result" or timestamp-unknown_since >= 1.2:
                break
            continue
        unknown_since = None
        if time.monotonic()-timestamp > .30:
            continue
        transform, gray = camera.align(frame)
        groups = tracker.observe(frame, timestamp, transform=transform)
        proposal, assessments = policy.decide(timestamp, groups, transform, gray)
        entry = {"t": round(timestamp-start, 3), "aligned": transform is not None,
                 "capture_ms": capture_ms, "capture_source": source, "assessments": assessments}
        continuations = [group for group in groups if group["source"] == "single_bar"]
        if continuations:
            entry["single_bar_groups"] = continuations
        report["samples"].append(entry)
        if not proposal:
            continue
        hud = recognize(image, viewport, *templates)
        spell = next((c for c in hud["cards"] if c["slot"] == HEAL_SLOT), None)
        if not spell or not spell["matches_reference"]:
            continue
        available = spell["remaining"]
        if available is None and any(COUNT_BITS(quantity_mask(image, card_position, dy=dy) ^ selected_one) <= 72
                                     for dy in (-1, 0, 1)):
            available = 1
        entry["quantity_before"] = available
        x, y = proposal["pixel"]
        if available != stock-len(attempts) or not (0 <= x < VIEWPORT[0] and 60 <= y < 810):
            continue
        point = [round(v*dst/src) for v, dst, src in zip((x, y), viewport, VIEWPORT)]
        attempt = {**proposal, "state": "dispatching", "point": point, "quantity_before": available,
                   "time": entry["t"]}
        attempts.append(attempt)
        persist()
        card_x, card_y = spell["center"]
        taps(adb, f"input tap {card_x} {card_y}; sleep 0.15; input tap {point[0]} {point[1]}")
        policy.record_cast(time.monotonic(), proposal)
        attempt["state"] = "tap_sent"
        persist()
        image.save(args.output / f"heal-{len(attempts)}-before.png")
    persist()


def run(adb, args):
    args.output.mkdir()
    templates = [normalized_image(p)[0] for p in (REFERENCE, BATTLE_REFERENCE, RESULT_REFERENCE)]
    _, viewport = normalized_image(args.reference / "pre.png")
    calibration = json.loads((args.reference / "calibration.json").read_text())
    profiles = json.loads((ROOT / "th19-defense-ranges.json").read_text())["defenses"]
    from ultralytics import YOLO

    model = YOLO(str(args.weights), task="detect")
    if args.ready:
        png, image = wait_for_phase(adb, templates, "pre_deployment")
    else:
        png, image = collect_predeployment(adb, templates, args.zoom_result, args.commands_open)
    (args.output / "pre.png").write_bytes(png)
    live_grid = projection_for_viewport(calibration, *image.size)
    layout = predict_layout(model, args.output / "pre.png", args.weights, .45, args.device, [960],
                            viewport_width=1000)
    (args.output / "layout.json").write_text(json.dumps(layout, indent=2)+"\n")
    with Image.open(args.output / "pre.png") as source:
        camera = BattleCamera(battlefield_frame(source.convert("RGB")))
    fire = DefenseFire(layout, profiles, live_grid, camera.reference)
    cells, bounds = outer_lawn_cells(image, live_grid, padding=2)
    if not cells:
        raise RuntimeError("No grass outside the red boundary")
    hall = max((d for d in layout["detections"] if d["class"] == "town_hall"),
               key=lambda d:d["confidence"])
    x1, _, x2, y2 = hall["bbox_xyxy"]
    layout_width, layout_height = (layout["viewport"][key] for key in ("width", "height"))
    target = live_grid.pixel_to_grid((x1+x2)*image.width/(2*layout_width),
                                    y2*image.height/layout_height)
    side, (dragon, queen), center, main = edge_opening(cells, bounds, target)
    hud = recognize(image, viewport, *templates)
    available = {card["slot"]: card for card in hud.get("cards", []) if card["matches_reference"]}
    expected = {"miner_squad": 4, "hog_squad": 2, "ice_golem": 2, "druid": 2, "ice_block": 2}
    if args.heal and (HEAL_SLOT not in available or not available[HEAL_SLOT]["remaining"]):
        raise RuntimeError("Expected a recognizable Healing Spell in card slot 11")
    if any(slot not in available for slot in CARDS.values()) or any(
        available[CARDS[code]]["remaining"] != count for code,count in expected.items()
    ):
        raise RuntimeError("Current cards or quantities differ from the configured army")

    def pixel(cell):
        x,y=live_grid.grid_to_pixel(cell[0]+.5,cell[1]+.5)
        return (round(x*viewport[0]/image.width),round(y*viewport[1]/image.height))

    stages = [
        ("dragon_flank", [("dragon_duke",[dragon])]),
        ("queen_flank", [("ice_golem",[queen]),("archer_queen",[queen]),("siege_barracks",[queen])]),
        ("main_wave", [("miner_squad",main),("hog_squad",main[1:3]),("grand_warden",[center]),
                        ("royal_champion",[center]),("druid",main[1:3])]),
    ]
    preview = Image.open(args.output/"pre.png").convert("RGB")
    draw=ImageDraw.Draw(preview)
    for name,cell,color in (("DRAGON",dragon,"#ffac4d"),("QUEEN",queen,"#6af6ad"),("MAIN",center,"#61d9ff")):
        x,y=pixel(cell)
        draw.ellipse((x-14,y-14,x+14,y+14),outline=color,width=4)
        draw.text((x+18,y-8),name,fill=color,stroke_width=2,stroke_fill="black")
    preview.save(args.output/"entry.png")
    report={"side": side,"grass_candidates":len(cells),"stages":[]}
    report_path=args.output/"run.json"

    def persist():
        report_path.write_text(json.dumps(report,indent=2)+"\n")

    for index,(name,commands) in enumerate(stages):
        stage={"name":name,"commands":[{"unit":unit,"slot":CARDS[unit],"points":[pixel(c) for c in points]}
                                        for unit,points in commands],"status":"preview"}
        report["stages"].append(stage)
        persist()
        if not args.execute:
            continue
        instructions=[]
        for command in stage["commands"]:
            slot=command["slot"]
            x,y=available[slot]["center"]
            instructions.extend((f"input tap {x} {y}", "sleep 0.15"))
            for x,y in command["points"]:
                instructions.extend((f"input tap {x} {y}", "sleep 0.12"))
        instruction_batch = "; ".join(instructions)
        if name == "main_wave" and args.heal:
            terrain = HealingMap(layout, profiles, live_grid, camera.reference)
            deploy_main_wave_with_healing(adb, args, templates, viewport, available[HEAL_SLOT],
                                          camera, terrain, layout, stage, image, instruction_batch, persist)
            continue
        stage["dispatch_started_monotonic_s"] = time.monotonic()
        taps(adb, instruction_batch)
        stage["dispatch_finished_monotonic_s"] = time.monotonic()
        if index == 0:
            deadline = time.monotonic()+5
            while True:
                first_frame = read_battle_frame(adb, templates, viewport)
                after, _, phase, _ = first_frame
                if phase == "battle":
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Dragon Duke deployment did not enter battle: {phase}")
            after.save(args.output / f"{name}.png")
        else:
            after_png, after = wait_for_phase(adb,templates,"battle")
            (args.output/f"{name}.png").write_bytes(after_png)
        stage_card_changes(stage, image, after, viewport, templates, persist)
        if index==1:
            time.sleep(args.funnel_wait)
        elif index==0:
            x, y = pixel(dragon)
            seed = tuple(value * dst / src for value, dst, src in zip((x, y), VIEWPORT, viewport))
            cast_dragon_ice_blocks(adb, args, templates, viewport, available[CARDS["ice_block"]]["center"],
                                   camera, fire, seed, first_frame, stage, persist)
        if index < len(stages)-1:
            _,_,phase=read_frame(adb,templates)
            if phase not in ("pre_deployment","battle"):
                raise RuntimeError(f"Cannot continue from {phase}")
    print(report_path,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--funnel-wait",type=float,default=12)
    parser.add_argument("--dragon-window",type=float,default=20,help="Seconds to observe the solo Dragon Duke")
    parser.add_argument("--ice-block-cooldown",type=float,default=7.5,
                        help="Minimum seconds between Ice Block casts")
    parser.add_argument("--weights",type=Path,default=ROOT / "models" / "realistic-defense20-yolo26s-r1.pt")
    parser.add_argument("--device",help="YOLO inference device")
    parser.add_argument("--ready",action="store_true",help="Use an already zoomed-out deployment screen")
    parser.add_argument("--commands-open", action="store_true", help="Start from the open Commands menu")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--heal", action="store_true", help="Heal the observed main wave in splash range or at low health")
    parser.add_argument("--zoom-result",type=Path,default=Path("D:/AI4COC/mumu-zoom-result.txt"))
    parser.add_argument("--adb",default="D:/MuMuPlayer/nx_main/adb.exe")
    parser.add_argument("--serial",default="127.0.0.1:16384")
    args=parser.parse_args()
    if args.heal and not args.execute:
        parser.error("--heal requires --execute")
    if not 0<=args.funnel_wait<=60:
        parser.error("Expected a funnel wait of 0 to 60 seconds")
    if not 1 <= args.dragon_window <= 40 or not 7 <= args.ice_block_cooldown <= 30:
        parser.error("Expected a 1–40 second observation window and a 7–30 second Ice Block cooldown")
    subprocess.run([args.adb,"connect",args.serial],check=True,timeout=10)
    run([args.adb,"-s",args.serial],args)


if __name__=="__main__":
    main()
