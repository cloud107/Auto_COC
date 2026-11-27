"""Stage 1 Live Runner: 8x44x44 Grid State + Potential-Field FSM + BC Trajectory Recorder."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from battle_annotator import AnnotatedBattleRecorder, draw_opening_yolo
from battle_grid_state import (
    BattlefieldGridEncoder,
    GuardianTracker,
    TrajectoryRecorder,
    detect_enemy_hero_bars,
    detect_firespitter_orientations,
    detect_xbow_modes,
)
from capture_realistic_layouts import read_frame, taps, wait_for_phase
from dragon_ice_block import BattleCamera, DefenseFire, DragonTracker, IceBlockPolicy, VIEWPORT, battlefield_frame
from export_yolo_layout import predict_layout
from grid_projection import projection_for_viewport
from healing_spell import HealingWaveTracker
from miner_hog_opening import (
    CARDS,
    cast_dragon_ice_blocks,
    collect_predeployment,
    healing_frames,
    read_battle_frame,
    stage_card_changes,
)
from potential_fsm import PotentialFieldFSM, SLOTS, score_entry_sides
from read_destruction import read_result
from recognize_hud import (
    BATTLE_REFERENCE,
    CARD_CENTERS,
    REFERENCE,
    RESULT_REFERENCE,
    normalized_image,
    recognize,
)

ROOT = Path(__file__).resolve().parent


def hero_hp_fraction(image: Image.Image, slot: int) -> float:
    """Measure a deployed hero's HUD HP bar fraction in [0.0, 1.0] (full bar = 37px strictly at y=719..724).

    Hero portraits dim/turn black once their ability is spent even while alive; the HP bar at
    y=719..724 (1000x800 normalized HUD) stays green/yellow/red (and flashes red during Phoenix!)
    while alive, and drops to 0.0 once truly KO'd (avoiding portrait top pixels at y>=725).
    """
    x = CARD_CENTERS[slot - 1]
    strip = np.asarray(image.crop((x - 20, 719, x + 20, 724)).convert("RGB"))
    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Neon green/yellow bars need high S/V to reject grass (S~159, V~138); red/flashing bars are never grass.
    green_yellow = (h >= 15) & (h <= 75) & (s >= 180) & (v >= 160)
    red_hp = ((h <= 15) | (h >= 165)) & (s >= 95) & (v >= 95)
    hp_px = green_yellow | red_hp
    cols = np.flatnonzero(hp_px.sum(axis=0) >= 2)
    if len(cols) < 2 or int(np.count_nonzero(hp_px)) < 4:
        return 0.0
    return min(1.0, len(cols) / 37.0)


def hero_is_alive(image: Image.Image, slot: int) -> bool:
    """Detect whether a deployed hero is alive via the neon green/yellow/red HP bar above their HUD card."""
    return hero_hp_fraction(image, slot) > 0.0


def spell_slot_available(image: Image.Image, slot: int) -> bool:
    """Return True if a spell card in the normalized 1000x800 HUD is still colored (unspent) vs grayscale (x0)."""
    x = CARD_CENTERS[slot - 1]
    crop = np.asarray(image.crop((x - 16, 732, x + 16, 782)).convert("RGB"))
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean()) >= 45.0 and float(hsv[:, :, 2].mean()) >= 120.0


def run_stage1(adb, args):
    args.output.mkdir(parents=True, exist_ok=True)
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
    norm_grid = projection_for_viewport(calibration, 1000, 800)
    # Dual-scale [960, 1280] pre-battle inference eliminates Guardian misses (~80ms on GPU)
    layout = predict_layout(
        model, args.output / "pre.png", args.weights, 0.22, args.device, [960, 1280], viewport_width=1000
    )
    (args.output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")

    with Image.open(args.output / "pre.png") as source:
        pre_pil = source.convert("RGB")
        pre_bgr = battlefield_frame(pre_pil)
    camera = BattleCamera(pre_bgr)

    # Build 8x44x44 encoder; probe Firespitter 150° cones AND X-Bow ground/air modes in pre-deployment
    encoder = BattlefieldGridEncoder(layout, profiles, norm_grid, pre_pil, camera.reference)
    full_grid = projection_for_viewport(calibration, *viewport)
    pre_rgb_arr = np.asarray(pre_pil)
    w_src, h_src = layout["viewport"]["width"], layout["viewport"]["height"]

    firespitter_orientations = []
    fs_dets = sorted(
        (d for d in layout.get("detections", []) if d["class"] == "firespitter" and d.get("confidence", 0.0) >= 0.45),
        key=lambda d: d["confidence"],
        reverse=True,
    )
    if fs_dets:
        try:
            x1, y1, x2, y2 = fs_dets[0]["bbox_xyxy"]
            fx = round((x1 + x2) * 0.5 * viewport[0] / w_src)
            fy = round((y1 + (y2 - y1) * 0.55) * viewport[1] / h_src)
            taps(adb, f"input tap {fx} {fy}")
            time.sleep(0.40)
            cone_png, _, cone_phase = read_frame(adb, templates)
            if cone_phase == "pre_deployment":
                cone_path = args.output / "pre_firespitter_cone.png"
                cone_path.write_bytes(cone_png)
                with Image.open(cone_path) as cone_src:
                    cone_rgb = np.asarray(cone_src.convert("RGB"))
                firespitter_orientations = detect_firespitter_orientations(
                    pre_rgb_arr, cone_rgb, layout, full_grid, encoder.buildings
                )
                print(f"[PRE] Firespitter cones: {firespitter_orientations}", flush=True)
        except Exception as exc:
            print(f"[PRE] Firespitter cone probe skipped: {exc}", flush=True)

    xbow_modes = []
    xb_dets = sorted(
        (d for d in layout.get("detections", []) if d["class"] == "x_bow" and d.get("confidence", 0.0) >= 0.45),
        key=lambda d: d["confidence"],
        reverse=True,
    )
    if xb_dets:
        try:
            x1, y1, x2, y2 = xb_dets[0]["bbox_xyxy"]
            xx = round((x1 + x2) * 0.5 * viewport[0] / w_src)
            xy = round((y1 + (y2 - y1) * 0.55) * viewport[1] / h_src)
            taps(adb, f"input tap {xx} {xy}")
            time.sleep(0.40)
            xb_png, _, xb_phase = read_frame(adb, templates)
            if xb_phase == "pre_deployment":
                xb_path = args.output / "pre_xbow_modes.png"
                xb_path.write_bytes(xb_png)
                with Image.open(xb_path) as xb_src:
                    xb_rgb = np.asarray(xb_src.convert("RGB"))
                xbow_modes = detect_xbow_modes(
                    pre_rgb_arr, xb_rgb, layout, full_grid, encoder.buildings
                )
                print(f"[PRE] X-Bow modes: {xbow_modes}", flush=True)
        except Exception as exc:
            print(f"[PRE] X-Bow mode probe skipped: {exc}", flush=True)

    if firespitter_orientations or xbow_modes:
        (args.output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")

    # Build DefenseFire AFTER pre-deployment probes so ground-only X-Bows are excluded & Firespitter cones respected!
    fire = DefenseFire(layout, profiles, live_grid, camera.reference)

    init_state = encoder.encode(0.0)
    plan = score_entry_sides(init_state["spatial"], encoder.buildings)
    guardian_tracker = (
        GuardianTracker(encoder.tile_to_viewport(encoder.guardian.tile), pre_bgr)
        if encoder.guardian is not None else None
    )

    # Save opening YOLO + 44x44 entry plan preview image
    opening_vis = draw_opening_yolo(pre_bgr, layout, plan, encoder, encoder.guardian)
    cv2.imwrite(str(args.output / "pre_yolo.jpg"), opening_vis)
    video_recorder = (
        AnnotatedBattleRecorder(args.output / "battle_annotated.mp4", fps=10.0)
        if not args.no_record_video else None
    )
    if video_recorder is not None:
        video_recorder.write_still(opening_vis, seconds=2.5)

    hud = recognize(image, viewport, *templates)
    available = {c["slot"]: c for c in hud.get("cards", []) if c["matches_reference"]}
    card_counts = {slot: (c["remaining"] or 1) for slot, c in available.items()}
    hero_states = [0, 0, 0, 0]  # [Dragon, Queen, Warden, RC]

    def tile_to_screen(cell, transform=None):
        vp = encoder.tile_to_viewport((cell[0] + 0.5, cell[1] + 0.5), transform=transform)
        return (round(float(vp[0]) * viewport[0] / VIEWPORT[0]), round(float(vp[1]) * viewport[1] / VIEWPORT[1]))

    report = {
        "mode": "stage1_potential_fsm",
        "weights": str(args.weights),
        "guardian": (
            {"name": encoder.guardian.name, "tile": [round(float(v), 2) for v in encoder.guardian.tile]}
            if encoder.guardian is not None else None
        ),
        "firespitters": firespitter_orientations,
        "x_bows": xbow_modes,
        "plan": {k: (list(v) if isinstance(v, tuple) else v) for k, v in plan.items()},
        "stages": [],
        "actions": [],
    }
    report_path = args.output / "run.json"

    def persist():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def build_tap_batch(cmd_list):
        instructions = []
        for cmd in cmd_list:
            slot = cmd["slot"]
            if slot not in available:
                continue
            sx, sy = available[slot]["center"]
            instructions.extend((f"input tap {sx} {sy}", "sleep 0.14"))
            for px, py in cmd["points"]:
                instructions.extend((f"input tap {px} {py}", "sleep 0.11"))
        return "; ".join(instructions)

    def make_stage(name, unit_cells):
        return {
            "name": name,
            "commands": [
                {"unit": u, "slot": CARDS[u], "points": [tile_to_screen(c) for c in cells]}
                for u, cells in unit_cells
            ],
            "status": "preview",
        }

    dragon_stage = make_stage("dragon_flank", [("dragon_duke", [plan["dragon_flank"]])])
    # Match human 3-star opening: 2 Ice Golems + Archer Queen + Siege Barracks at 6 o'clock corner (t=+6s in parallel)
    queen_stage = make_stage("queen_flank", [
        ("ice_golem", [plan["queen_flank"], plan["queen_flank"]]),
        ("archer_queen", [plan["queen_flank"]]),
        ("siege_barracks", [plan["queen_flank"]]),
    ])
    main_stage = make_stage("main_wave", [
        ("miner_squad", plan["main_line"]),
        ("hog_squad", plan["main_line"][1:3]),
        ("grand_warden", [plan["center"]]),
        ("royal_champion", [plan["center"]]),
        ("druid", plan["main_line"][1:3]),
    ])
    report["stages"] = [dragon_stage, queen_stage, main_stage]
    persist()

    if not args.execute:
        if video_recorder is not None:
            video_recorder.close()
        print(report_path, flush=True)
        return

    # --- PHASE 1: PARALLEL FUNNEL (Dragon @ t=0s + 2x Ice Golem & Queen @ t=6s + live Ice Blocks) ---
    dragon_stage["dispatch_started_monotonic_s"] = time.monotonic()
    taps(adb, build_tap_batch(dragon_stage["commands"]))
    dragon_stage["dispatch_finished_monotonic_s"] = time.monotonic()
    battle_started_at = dragon_stage["dispatch_started_monotonic_s"]
    dragon_stage["status"] = "sent"
    hero_states[0] = 1

    deadline = time.monotonic() + 5
    while True:
        first_frame = read_battle_frame(adb, templates, viewport)
        after_dragon, _, phase, _ = first_frame
        if phase == "battle":
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Dragon deployment did not enter battle: {phase}")
    after_dragon.save(args.output / "dragon_flank.png")
    stage_card_changes(dragon_stage, image, after_dragon, viewport, templates, persist)

    sx, sy = tile_to_screen(plan["dragon_flank"])
    dragon_seed = (sx * VIEWPORT[0] / viewport[0], sy * VIEWPORT[1] / viewport[1])
    dragon_tracker = DragonTracker(dragon_seed, pre_bgr=pre_bgr)
    ice_policy = IceBlockPolicy(args.ice_block_cooldown)
    ice_slot_xy = available[CARDS["ice_block"]]["center"]
    ice_casts = []
    dragon_stage["ice_block"] = {"casts": ice_casts}

    df = np.asarray(plan["dragon_flank"], dtype=float)
    dragon_last_tile = df + 5.0 * (np.array([22.0, 22.0]) - df) / max(1.0, float(np.linalg.norm(np.array([22.0, 22.0]) - df)))
    dragon_still_alive = True
    # Phoenix pet revives Dragon Duke ~1s after 1st death; track death transitions so Revive waits for 2nd (true) death!
    dragon_prev_alive = True
    dragon_death_count = 0
    dragon_gray_since: float | None = None
    dragon_ready_to_revive = False

    def update_dragon_life(hud_img, t_battle: float):
        nonlocal dragon_still_alive, dragon_prev_alive, dragon_death_count, dragon_gray_since, dragon_ready_to_revive
        if t_battle < 4.0:
            return
        # While Dragon Duke is alive OR in Phoenix invulnerability, the HUD HP bar either stays colored
        # or flashes red every ~0.5-1.0s; once truly KO'd, the bar stays 0.0 continuously with NO red flash!
        alive_now = hero_hp_fraction(hud_img, SLOTS["dragon_duke"]) > 0.0
        if alive_now:
            dragon_still_alive = True
            dragon_prev_alive = True
            dragon_gray_since = None
        else:
            if dragon_gray_since is None:
                dragon_gray_since = t_battle
            gray_dur = t_battle - dragon_gray_since
            # Continuously 0.0 (no green/yellow/red flash at all) for >= 2.5s means Phoenix is over & truly KO'd!
            if gray_dur >= 2.5:
                dragon_still_alive = False
                dragon_ready_to_revive = True

    queen_deployed = False
    queen_deployed_at: float | None = None
    queen_hp_hist: list[tuple[float, float]] = []
    queen_zero_since: float | None = None

    def maybe_trigger_queen_ability(hud_img, frame_bgr, t_battle: float, t_log: float, phase_label: str) -> bool:
        nonlocal queen_hp_hist, queen_zero_since
        if not queen_deployed or hero_states[1] != 1 or queen_deployed_at is None:
            return False
        if t_battle - queen_deployed_at < 1.5:
            return False
        q_slot = SLOTS["archer_queen"]
        q_info = available.get(q_slot)
        if q_info is None:
            return False
        q_hp = hero_hp_fraction(hud_img, q_slot)
        if q_hp <= 0.0:
            if queen_hp_hist:
                if queen_zero_since is None:
                    queen_zero_since = t_battle
                elif t_battle - queen_zero_since >= 1.2:
                    hero_states[1] = 2  # already KO'd or auto-ability spent
            return False
        queen_zero_since = None
        queen_hp_hist.append((t_battle, q_hp))
        queen_hp_hist = [(t, v) for t, v in queen_hp_hist if t_battle - t <= 1.6]
        if len(queen_hp_hist) < 2:
            return False
        peak_hp = max(v for _, v in queen_hp_hist)
        # Use max of last 2 readings to filter single-frame HUD flicker
        cur_hp = max(queen_hp_hist[-1][1], queen_hp_hist[-2][1])
        drop = peak_hp - cur_hp
        reason = None
        if len(queen_hp_hist) >= 3 and drop >= 0.20 and cur_hp <= 0.78:
            reason = f"queen_rapid_drop_{int(round(drop * 100))}pct"
        elif cur_hp <= 0.38 and peak_hp >= 0.45:
            reason = f"queen_low_hp_{int(round(cur_hp * 100))}pct"
        if reason is None:
            return False
        qx, qy = q_info["center"]
        taps(adb, f"input tap {qx} {qy}")
        hero_states[1] = 2
        ev = {
            "t": round(t_log, 3),
            "slot": q_slot,
            "grid": None,
            "kind": "hero_ability",
            "reason": reason,
            "phase": phase_label,
        }
        report["actions"].append(ev)
        persist()
        if video_recorder is not None:
            video_recorder.write_banner_frame(
                frame_bgr,
                f"DECISION +{t_battle:.1f}s: QUEEN ABILITY ({reason})",
                f"hp={int(round(cur_hp * 100))}% (peak={int(round(peak_hp * 100))}%)",
                seconds=0.8,
            )
        return True

    funnel_log = {"samples": []}

    with closing(healing_frames(adb, templates, viewport, funnel_log)) as fframes:
        for fs in fframes:
            elapsed_b = time.monotonic() - battle_started_at
            if elapsed_b >= args.main_wave_at or fs.phase != "battle":
                break
            if time.monotonic() - fs.timestamp > 0.35:
                continue

            # Deploy 2 Ice Golems + Archer Queen at 6 o'clock corner at t >= 6.0s (parallel funnel!)
            if not queen_deployed and elapsed_b >= args.queen_delay:
                queen_stage["dispatch_started_monotonic_s"] = time.monotonic()
                taps(adb, build_tap_batch(queen_stage["commands"]))
                queen_stage["dispatch_finished_monotonic_s"] = time.monotonic()
                queen_stage["status"] = "sent"
                hero_states[1] = 1
                card_counts[SLOTS["siege_barracks"]] = 0
                card_counts[SLOTS["ice_golem"]] = 0
                queen_deployed = True
                queen_deployed_at = time.monotonic() - battle_started_at
                persist()
                if video_recorder is not None:
                    video_recorder.write_banner_frame(
                        fs.frame,
                        f"FUNNEL +{elapsed_b:.1f}s: 2x ICE GOLEM + QUEEN + SIEGE @ 6 O'CLOCK",
                        f"queen_flank={list(plan['queen_flank'])} (parallel with Dragon)",
                        seconds=0.8,
                    )
                continue

            # Monitor Queen HUD HP for rapid drop during funnel
            if maybe_trigger_queen_ability(fs.image, fs.frame, elapsed_b, -max(0.0, args.main_wave_at - elapsed_b), "FUNNEL"):
                continue

            # Track Dragon Duke, Guardian & outer non-defense funnel clearance continuously during funnel!
            ftr, fgray = camera.align(fs.frame)
            if ftr is not None and fgray is not None:
                encoder.update_alive_buildings(elapsed_b, ftr, fgray)
            if guardian_tracker is not None and ftr is not None:
                guardian_tracker.observe(fs.frame, elapsed_b, transform=ftr)
            update_dragon_life(fs.image, elapsed_b)
            dragon_hud_hp = hero_hp_fraction(fs.image, SLOTS["dragon_duke"])
            obs = None
            if dragon_still_alive and ftr is not None:
                obs = dragon_tracker.observe(fs.frame, ftr, fs.timestamp, hud_hp_frac=dragon_hud_hp)
                if obs is not None:
                    inv_f = cv2.invertAffineTransform(ftr)
                    dragon_last_tile = encoder.viewport_to_tile(inv_f @ np.r_[obs.position, 1.0])

            # Evaluate Ice Block cast (up to 2 Ice Blocks); drop on Dragon Duke's GROUND SHADOW (+28px below flying sprite)
            if dragon_still_alive and len(ice_casts) < 2 and obs is not None and ftr is not None:
                confirmed, possible = fire.estimate(obs.position, ftr, fgray, fs.timestamp)
                should_cast, score = ice_policy.evaluate(
                    fs.timestamp, obs.health_width, confirmed, hud_hp_frac=dragon_hud_hp
                )
                if should_cast:
                    shadow_pos = (float(obs.position[0]), min(float(VIEWPORT[1] - 10), float(obs.position[1]) + 28.0))
                    px = round(shadow_pos[0] * viewport[0] / VIEWPORT[0])
                    py = round(shadow_pos[1] * viewport[1] / VIEWPORT[1])
                    taps(adb, f"input tap {ice_slot_xy[0]} {ice_slot_xy[1]}; sleep 0.14; input tap {px} {py}")
                    ice_policy.record_attempt(fs.timestamp)
                    ev = {
                        "number": len(ice_casts) + 1,
                        "t": round(elapsed_b, 2),
                        "point": [px, py],
                        "health_width": obs.health_width,
                        "score": score,
                    }
                    ice_casts.append(ev)
                    card_counts[SLOTS["ice_block"]] = max(0, card_counts.get(SLOTS["ice_block"], 0) - 1)
                    persist()
                    if video_recorder is not None:
                        video_recorder.write_banner_frame(
                            fs.frame,
                            f"DECISION +{elapsed_b:.1f}s: ICE BLOCK #{ev['number']} (Dragon Duke)",
                            f"hp_w={obs.health_width} | air_fire_score={score}",
                            target_px=shadow_pos,
                            seconds=1.2,
                        )
                    continue

            if video_recorder is not None:
                rem_s = max(0.0, args.main_wave_at - elapsed_b)
                video_recorder.write_banner_frame(
                    fs.frame,
                    f"PARALLEL FUNNEL +{elapsed_b:.1f}s | main wave in {rem_s:.1f}s | ice={len(ice_casts)}/2",
                    f"Dragon {'alive' if dragon_still_alive else 'fallen'} @ {[round(float(v),1) for v in dragon_last_tile]} | Queen {'clearing' if queen_deployed else 'waiting'}",
                    target_px=(obs.position if obs is not None else None),
                    seconds=0.1,
                )

    # --- PHASE 2 & 3: MAIN_PUSH (@ t=34s) + SIEGE BARRACKS (@ +6s) + 10Hz CLOSED_LOOP_MICRO + CLEANUP ---
    fsm = PotentialFieldFSM(
        funnel_wait_s=args.funnel_wait,
        warden_min_delay_s=args.warden_min_delay,
        enable_freeze=args.enable_freeze,
        enable_totem=args.enable_totem,
        enable_warden_ability=args.enable_warden_ability,
        enable_rc_ability=args.enable_rc_ability,
        enable_cleanup=args.enable_cleanup,
        enable_revive=args.enable_revive,
    )
    fsm.phase = "CLOSED_LOOP_MICRO"
    fsm.entry_center = tuple(plan["center"])
    recorder = TrajectoryRecorder() if args.record_bc else None
    stream_report = {"samples": [], "casts": []}
    main_stage["fsm_micro"] = stream_report
    main_batch = build_tap_batch(main_stage["commands"])

    def dispatch_main():
        taps(adb, main_batch)
        return time.monotonic()

    result_image = None
    with closing(healing_frames(adb, templates, viewport, stream_report)) as frames:
        before = next(frames)
        if before.phase != "battle":
            raise RuntimeError(f"Cannot deploy main wave from {before.phase}")
        started = time.monotonic()
        main_stage["dispatch_started_monotonic_s"] = started
        main_stage["status"] = "sent"
        tracker = HealingWaveTracker(started, main_stage["commands"], layout, pre_bgr=pre_bgr)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(dispatch_main)
            while not pending.done():
                sample = next(frames)
                if sample.timestamp >= started and sample.phase == "battle":
                    tr, _ = camera.align(sample.frame)
                    # Do NOT search for Grand Warden before dispatch_main() has finished placing him!
                    tracker.observe(sample.frame, sample.timestamp, transform=tr, warden_alive=False)
                    if video_recorder is not None:
                        video_recorder.write_banner_frame(
                            sample.frame,
                            f"MAIN PUSH DEPLOYING  |  +{sample.timestamp - started:.1f}s",
                            f"Miners + Hogs + Warden + RC + Druids @ {plan['side']}",
                            seconds=0.1,
                        )
            finished = pending.result()
        main_stage["dispatch_finished_monotonic_s"] = finished
        hero_states[2] = 1  # Warden ready
        hero_states[3] = 1  # Royal Champion ready
        for u in ("miner_squad", "hog_squad", "druid"):
            card_counts[SLOTS[u]] = 0

        unknown_since = None
        last_revive_attempt_at = -99.0
        revive_pending_hero: int | None = None
        revive_attempts = 0
        for sample in frames:
            ts, img, frame, phase, capture_ms, source = sample
            if ts - finished >= 150:
                break
            if phase != "battle":
                if phase == "result":
                    result_image = img
                    break
                if unknown_since is None:
                    unknown_since = ts
                if ts - unknown_since >= 1.5:
                    break
                continue
            unknown_since = None
            if time.monotonic() - ts > 0.30:
                continue

            tr, gray = camera.align(frame)
            elapsed_main = ts - started
            warden_hud_hp = 1.0 if elapsed_main < 3.0 else hero_hp_fraction(img, SLOTS["grand_warden"])
            rc_hud_hp = 1.0 if elapsed_main < 3.0 else hero_hp_fraction(img, SLOTS["royal_champion"])
            warden_alive = warden_hud_hp > 0.0
            if not warden_alive and hero_states[2] == 1:
                hero_states[2] = 2
            if elapsed_main >= 3.0 and rc_hud_hp <= 0.0 and hero_states[3] == 1:
                hero_states[3] = 2
            groups = tracker.observe(frame, ts, transform=tr, warden_alive=warden_alive)

            # Continuously track the walking Guardian using unified battle clock (stops cleanly upon Guardian KO!)
            live_guardian_tile = None
            guardian_awake = False
            if guardian_tracker is not None:
                g_feet_ref, guardian_awake = guardian_tracker.observe(frame, ts - battle_started_at, transform=tr)
                if g_feet_ref is not None:
                    live_guardian_tile = encoder.viewport_to_tile(g_feet_ref)
                elif encoder.guardian is not None:
                    encoder.guardian.alive = False

            # Scan for defending Guardian / enemy hero purple bars around BOTH main army AND Guardian shrine
            enemy_tiles = []
            scan_centers = []
            if groups:
                scan_centers.append(groups[0]["center"])
            if encoder.guardian is not None:
                scan_centers.append(encoder.tile_to_viewport(encoder.guardian.tile, transform=tr))
            if scan_centers:
                lo = np.rint(np.min(scan_centers, axis=0) - 180).astype(int)
                hi = np.rint(np.max(scan_centers, axis=0) + 180).astype(int)
                roi = (max(0, lo[0]), max(75, lo[1]), min(VIEWPORT[0], hi[0]), min(820, hi[1]))
                inv = cv2.invertAffineTransform(tr) if tr is not None else None
                for ex, ey in detect_enemy_hero_bars(frame, pre_bgr, roi):
                    pt = inv @ np.r_[ex, ey, 1.0] if inv is not None else np.array([ex, ey])
                    enemy_tiles.append(encoder.viewport_to_tile(pt))

            # Continue monitoring Queen HUD HP for rapid drop during main push
            maybe_trigger_queen_ability(img, frame, ts - battle_started_at, elapsed_main, fsm.phase)

            # Continue tracking Dragon Duke & casting any remaining Ice Block during main push
            # ONLY when Dragon Duke is separated from the main army (never freeze the main army!).
            update_dragon_life(img, ts - battle_started_at)
            dragon_hud_hp_main = hero_hp_fraction(img, SLOTS["dragon_duke"])
            if dragon_still_alive and card_counts.get(SLOTS["ice_block"], 0) > 0 and tr is not None:
                d_obs = dragon_tracker.observe(frame, tr, ts, hud_hp_frac=dragon_hud_hp_main)
                if d_obs is not None:
                    inv_d = cv2.invertAffineTransform(tr)
                    dragon_last_tile = encoder.viewport_to_tile(inv_d @ np.r_[d_obs.position, 1.0])
                    near_main = False
                    if groups:
                        g_ref = inv_d @ np.r_[groups[0]["center"], 1.0]
                        g_tile = encoder.viewport_to_tile(g_ref)
                        near_main = float(np.linalg.norm(dragon_last_tile - g_tile)) < 7.0
                    if not near_main:
                        confirmed_d, _ = fire.estimate(d_obs.position, tr, gray, ts)
                        should_ice, ice_score = ice_policy.evaluate(
                            ts, d_obs.health_width, confirmed_d, hud_hp_frac=dragon_hud_hp_main
                        )
                        if should_ice:
                            shadow_pos = (float(d_obs.position[0]), min(float(VIEWPORT[1] - 10), float(d_obs.position[1]) + 28.0))
                            ipx = round(shadow_pos[0] * viewport[0] / VIEWPORT[0])
                            ipy = round(shadow_pos[1] * viewport[1] / VIEWPORT[1])
                            taps(adb, f"input tap {ice_slot_xy[0]} {ice_slot_xy[1]}; sleep 0.12; input tap {ipx} {ipy}")
                            ice_policy.record_attempt(ts)
                            ev = {
                                "number": len(ice_casts) + 1,
                                "t": round(ts - battle_started_at, 2),
                                "point": [ipx, ipy],
                                "health_width": d_obs.health_width,
                                "score": ice_score,
                            }
                            ice_casts.append(ev)
                            card_counts[SLOTS["ice_block"]] = max(0, card_counts.get(SLOTS["ice_block"], 0) - 1)
                            persist()
                            if video_recorder is not None:
                                video_recorder.write_banner_frame(
                                    frame,
                                    f"DECISION +{elapsed_main:.1f}s: ICE BLOCK #{ev['number']} (Dragon Duke)",
                                    f"hp_w={d_obs.health_width} | air_fire_score={ice_score}",
                                    target_px=shadow_pos,
                                    seconds=0.9,
                                )

            # Verify whether a previous Revive Spell attempt was consumed (slot 16 turned grayscale x0).
            # Cap retries at 2 (with 3.5s spacing) so Revive NEVER spams or delays other battle micro!
            if revive_pending_hero is not None and (ts - last_revive_attempt_at) >= 0.9:
                if not spell_slot_available(img, SLOTS["revive_spell"]):
                    card_counts[SLOTS["revive_spell"]] = 0
                    if revive_pending_hero == SLOTS["dragon_duke"]:
                        dragon_ready_to_revive = False
                        dragon_still_alive = True
                    revive_pending_hero = None
                elif (ts - last_revive_attempt_at) >= 3.5:
                    revive_pending_hero = None
                    if revive_attempts >= 2:
                        # Stop retrying after 2 failed attempts (e.g. hero wasn't actually KO'd); wait for next real KO.
                        dragon_ready_to_revive = False
                        dragon_gray_since = None
                        revive_attempts = 0

            # Revive Spell: reserve strictly for Dragon Duke (once red-flash stops for >=2.5s = true KO);
            # in endgame, if Dragon Duke is safe, spend Revive on ANY fallen hero (RC -> Queen -> Warden) to maximize value!
            fallen_target = None
            if (
                args.enable_revive
                and revive_pending_hero is None
                and card_counts.get(SLOTS["revive_spell"], 0) > 0
                and spell_slot_available(img, SLOTS["revive_spell"])
            ):
                if dragon_ready_to_revive:
                    fallen_target = SLOTS["dragon_duke"]
                else:
                    alive_defs = sum(1 for b in encoder.buildings if b.alive)
                    total_defs = max(1, len(encoder.buildings))
                    cleared_frac = 1.0 - (alive_defs / total_defs)
                    endgame_revive = (
                        dragon_still_alive
                        and (
                            (elapsed_main >= 35.0 and (cleared_frac >= 0.72 or alive_defs <= 5))
                            or elapsed_main >= 65.0
                        )
                    )
                    if endgame_revive:
                        if rc_hud_hp <= 0.0:
                            fallen_target = SLOTS["royal_champion"]
                        elif not hero_is_alive(img, SLOTS["archer_queen"]):
                            fallen_target = SLOTS["archer_queen"]
                        elif not warden_alive:
                            fallen_target = SLOTS["grand_warden"]

            state = encoder.encode(
                elapsed_main,
                transform=tr,
                gray=gray,
                frame_bgr=frame,
                army_groups=groups,
                active_spells=fsm.active_spells,
                card_counts=card_counts,
                hero_states=hero_states,
            )
            hero_hps = {
                SLOTS["dragon_duke"]: dragon_hud_hp_main,
                SLOTS["grand_warden"]: warden_hud_hp,
                SLOTS["royal_champion"]: rc_hud_hp,
            }
            act = fsm.step(
                state, card_counts, hero_states, groups,
                enemy_hero_tiles=enemy_tiles, fallen_hero_tile=fallen_target,
                live_guardian_tile=live_guardian_tile, guardian_awake=guardian_awake,
                hero_hps=hero_hps,
            )
            if recorder is not None:
                recorder.add(state, act)

            target_vp = None
            if act.slot != 0:
                slot_info = available.get(act.slot)
                if slot_info is not None:
                    sx, sy = slot_info["center"]
                    if act.kind == "hero_ability":
                        taps(adb, f"input tap {sx} {sy}")
                        if act.slot == SLOTS["grand_warden"]:
                            hero_states[2] = 2
                        elif act.slot == SLOTS["royal_champion"]:
                            hero_states[3] = 2
                    elif act.kind == "revive_hud" and act.grid is not None:
                        hero_slot_id = int(act.grid[0])
                        hero_info = available.get(hero_slot_id)
                        if hero_info is not None:
                            hx, hy = hero_info["center"]
                            taps(adb, f"input tap {sx} {sy}; sleep 0.18; input tap {hx} {hy}")
                            last_revive_attempt_at = ts
                            revive_pending_hero = hero_slot_id
                            revive_attempts += 1
                    elif act.grid is not None:
                        target_vp = encoder.tile_to_viewport((act.grid[0] + 0.5, act.grid[1] + 0.5), transform=tr)
                        px, py = tile_to_screen(act.grid, transform=tr)
                        tap_gap = "0.08" if act.slot == SLOTS["totem_spell"] else "0.14"
                        taps(adb, f"input tap {sx} {sy}; sleep {tap_gap}; input tap {px} {py}")
                        card_counts[act.slot] = max(0, card_counts.get(act.slot, 0) - 1)
                    if video_recorder is not None:
                        video_recorder.add_decision(elapsed_main, act, target_px=target_vp)
                    report["actions"].append({"t": round(elapsed_main, 3), **act.to_dict()})
                    persist()

            if video_recorder is not None:
                video_recorder.write_step(
                    frame, elapsed_main, fsm.phase, groups, encoder,
                    transform=tr, active_spells=fsm.active_spells,
                    live_guardian_tile=live_guardian_tile,
                )

    if recorder is not None:
        saved = recorder.save(args.output / "bc-trajectory.npz")
        report["bc_steps_recorded"] = saved

    # Wait for the result screen AND let the animated destruction counter settle (3s) before OCR
    if result_image is None:
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            _, cand_img, cand_phase = read_frame(adb, templates)
            if cand_phase == "result":
                result_image = cand_img
                break
            time.sleep(1.5)
    if result_image is not None:
        time.sleep(3.0)
        _, settled_img, settled_phase = read_frame(adb, templates)
        if settled_phase == "result":
            result_image = settled_img
        result_image.save(args.output / "result.png")
        try:
            report["final_destruction"] = read_result(result_image, ROOT)
        except Exception as exc:
            report["final_destruction_error"] = str(exc)
        if video_recorder is not None:
            res_bgr = cv2.cvtColor(np.asarray(result_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            video_recorder.write_still(res_bgr, seconds=2.5)
    if video_recorder is not None:
        video_recorder.close()
    persist()
    print(report_path, flush=True)


def _load_user_config(config_path: Path | None) -> dict:
    path = config_path or (ROOT / "config.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    cfg = _load_user_config(pre_args.config)

    default_ref = Path(cfg["reference"]) if "reference" in cfg else (ROOT / "captures" / "yolo26s-self-loop")
    default_weights = Path(cfg["weights"]) if "weights" in cfg else (ROOT / "models" / "realistic-defense24-yolo26m-self-20261125.pt")
    default_adb = cfg.get("adb", "D:/MuMuPlayer/nx_main/adb.exe")
    default_serial = cfg.get("serial", "127.0.0.1:16384")
    default_device = cfg.get("device", "auto")

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.add_argument("reference", nargs="?", type=Path, default=default_ref, help="Calibration reference dir (default: captures/yolo26s-self-loop)")
    parser.add_argument("output", nargs="?", type=Path, default=None, help="Output directory (default: runs/battle-YYYYMMDD-HHMMSS)")
    parser.add_argument("--funnel-wait", type=float, default=12.0, help="Unused legacy arg kept for CLI compatibility")
    parser.add_argument("--queen-delay", type=float, default=6.0, help="Seconds after Dragon Duke to deploy 2x Ice Golem + Queen in parallel")
    parser.add_argument("--main-wave-at", type=float, default=34.0, help="Deploy main_wave at this elapsed battle time (34s = 2:26 left, matching 3-star template)")
    parser.add_argument("--dragon-window", type=float, default=30.0)
    parser.add_argument("--ice-block-cooldown", type=float, default=7.5)
    parser.add_argument("--warden-min-delay", type=float, default=10.0, help="Minimum seconds after main_wave before Warden Tome can trigger")
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--device", default=default_device, help="Inference device: auto, 0, cpu (default: auto)")
    parser.add_argument("--ready", action="store_true", help="Already in pre-deployment screen (skip auto-entering battle)")
    parser.add_argument("--commands-open", action="store_true")
    parser.add_argument("--full", action="store_true", help="Shorthand for --execute plus all spell/hero/cleanup flags")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-freeze", action="store_true")
    parser.add_argument("--enable-totem", action="store_true")
    parser.add_argument("--enable-warden-ability", action="store_true")
    parser.add_argument("--enable-rc-ability", action="store_true")
    parser.add_argument("--enable-cleanup", action="store_true")
    parser.add_argument("--enable-revive", action="store_true")
    parser.add_argument("--record-bc", action="store_true", help="Save 10Hz BC dataset to bc-trajectory.npz")
    parser.add_argument("--no-record-video", action="store_true", help="Disable battle_annotated.mp4 recording")
    parser.add_argument("--zoom-result", type=Path, default=ROOT / "mumu-zoom-result.txt")
    parser.add_argument("--adb", default=default_adb, help="Path to MuMu adb executable")
    parser.add_argument("--serial", default=default_serial, help="MuMu ADB serial (e.g. 127.0.0.1:16384)")
    args = parser.parse_args()

    if args.full:
        args.execute = True
        args.enable_freeze = True
        args.enable_totem = True
        args.enable_warden_ability = True
        args.enable_rc_ability = True
        args.enable_cleanup = True
        args.enable_revive = True

    if args.output is None:
        out_root = Path(cfg["output_dir"]) if "output_dir" in cfg else (ROOT / "runs")
        args.output = out_root / time.strftime("battle-%Y%m%d-%H%M%S")

    args.device = _resolve_device(args.device)
    if not (Path(args.adb).exists() or shutil.which(args.adb)):
        raise SystemExit(
            f"ADB executable not found: {args.adb}\n"
            "Please copy config.example.json to config.json and set 'adb' & 'serial', "
            "or pass --adb <path-to-adb> --serial <host:port>."
        )

    subprocess.run([args.adb, "connect", args.serial], check=True, timeout=10)
    run_stage1([args.adb, "-s", args.serial], args)


if __name__ == "__main__":
    main()
