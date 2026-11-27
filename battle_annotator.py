"""Render annotated battle videos with opening YOLO overlay and real-time decision callouts."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from battle_grid_state import CORE_DEFENSE_CLASSES, GUARDIAN_CLASSES
from dragon_ice_block import VIEWPORT

SLOT_LABELS = {
    4: "ARCHER (Cleanup)",
    8: "QUEEN ABILITY",
    9: "WARDEN TOME",
    10: "RC ABILITY",
    11: "HEAL SPELL",
    12: "FREEZE SPELL",
    13: "TOTEM SPELL",
    14: "ICE BLOCK",
    15: "POISON SPELL",
    16: "REVIVE SPELL",
}

TRACK_COLORS = {
    "health_bars": (90, 255, 110),
    "single_bar": (70, 230, 255),
    "motion_only": (255, 130, 240),
    "warden_anchor": (40, 200, 255),
    "predicted": (180, 180, 180),
}


def draw_opening_yolo(pre_bgr: np.ndarray, layout: dict, plan: dict, encoder, guardian=None) -> np.ndarray:
    """Draw YOLO building boxes, Guardian highlight, and 44x44 entry plan on pre.png (1200x960)."""
    vis = pre_bgr.copy()
    if vis.shape[:2] != VIEWPORT[::-1]:
        vis = cv2.resize(vis, VIEWPORT, interpolation=cv2.INTER_AREA)
    w_src, h_src = layout["viewport"]["width"], layout["viewport"]["height"]
    sx, sy = VIEWPORT[0] / w_src, VIEWPORT[1] / h_src
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw detected directional Firespitter cones underneath boxes so they're easy to inspect
    cone_overlay = vis.copy()
    has_cones = False
    fs_summary: list[str] = []
    for b in getattr(encoder, "buildings", []):
        if b.facing_rad is None or b.cone_half_angle_rad is None:
            continue
        has_cones = True
        deg = round(math.degrees(b.facing_rad))
        fs_summary.append(f"{deg:+d}deg")
        r_cone = b.outer_range if b.outer_range > 0 else 16.0
        pts = [encoder.tile_to_viewport(b.tile)]
        for ang in np.linspace(b.facing_rad - b.cone_half_angle_rad, b.facing_rad + b.cone_half_angle_rad, 31):
            pts.append(encoder.tile_to_viewport((b.tile[0] + r_cone * math.cos(ang), b.tile[1] + r_cone * math.sin(ang))))
        poly = np.rint(np.asarray(pts, dtype=float)).astype(np.int32)
        cv2.fillPoly(cone_overlay, [poly], (30, 120, 255))
        cv2.polylines(vis, [poly], True, (60, 180, 255), 2, cv2.LINE_AA)
        p0 = tuple(poly[0])
        p_mid = tuple(np.rint(encoder.tile_to_viewport((
            b.tile[0] + 6.5 * math.cos(b.facing_rad),
            b.tile[1] + 6.5 * math.sin(b.facing_rad),
        ))).astype(int))
        cv2.arrowedLine(vis, p0, p_mid, (40, 230, 255), 2, cv2.LINE_AA, tipLength=0.25)
    if has_cones:
        vis = cv2.addWeighted(cone_overlay, 0.22, vis, 0.78, 0.0)

    # Draw zero-label detected non-defense buildings (Ch 3) as subtle green-cyan markers
    for nb in getattr(encoder, "non_defense_buildings", []):
        npt = tuple(np.rint(encoder.tile_to_viewport(nb.tile)).astype(int))
        cv2.circle(vis, npt, 7, (180, 255, 160), 1, cv2.LINE_AA)

    for d in layout.get("detections", []):
        cls = d["class"]
        conf = d.get("confidence", 0.0)
        if cls not in GUARDIAN_CLASSES and conf < 0.45:
            continue
        x1, y1, x2, y2 = d["bbox_xyxy"]
        p1, p2 = (round(x1 * sx), round(y1 * sy)), (round(x2 * sx), round(y2 * sy))
        if cls == "town_hall":
            col, thick = (0, 215, 255), 3
        elif cls in GUARDIAN_CLASSES:
            col, thick = (255, 40, 220), 3
        elif cls in CORE_DEFENSE_CLASSES:
            col, thick = (40, 140, 255), 2
        else:
            col, thick = (200, 200, 90), 1
        cv2.rectangle(vis, p1, p2, col, thick)
        if cls in GUARDIAN_CLASSES or cls == "town_hall" or cls in {"monolith", "scattershot", "inferno_tower", "firespitter", "x_bow"}:
            fdeg = d.get("facing_deg")
            xmode = d.get("mode")
            if fdeg is not None:
                tag = f"{cls} {conf:.2f} [{fdeg:+d}deg]"
            elif xmode is not None:
                tag = f"x_bow [{'G' if xmode == 'ground' else 'A+G'}]"
            else:
                tag = f"{cls} {conf:.2f}"
            cv2.putText(vis, tag, (p1[0], max(18, p1[1] - 4)), font, 0.44, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(vis, tag, (p1[0], max(18, p1[1] - 4)), font, 0.44, col, 1, cv2.LINE_AA)

    # Draw entry plan markers
    for label, key, col in (
        ("DRAGON", "dragon_flank", (40, 180, 255)),
        ("QUEEN", "queen_flank", (110, 255, 160)),
        ("MAIN", "center", (255, 220, 60)),
    ):
        cell = plan[key]
        pt = tuple(np.rint(encoder.tile_to_viewport((cell[0] + 0.5, cell[1] + 0.5))).astype(int))
        cv2.circle(vis, pt, 12, (15, 15, 15), 4, cv2.LINE_AA)
        cv2.circle(vis, pt, 12, col, 2, cv2.LINE_AA)
        cv2.putText(vis, label, (pt[0] + 14, pt[1] + 5), font, 0.58, (15, 15, 15), 4, cv2.LINE_AA)
        cv2.putText(vis, label, (pt[0] + 14, pt[1] + 5), font, 0.58, col, 2, cv2.LINE_AA)

    # Header summary bar
    cv2.rectangle(vis, (12, 38), (980, 108), (20, 20, 20), -1)
    g_txt = f"{guardian.name} @ [{guardian.tile[0]:.1f},{guardian.tile[1]:.1f}]" if guardian is not None else "None"
    fs_txt = f"  |  Firespitter cones: {', '.join(fs_summary)}" if fs_summary else ""
    line1 = f"OPENING YOLO & 44x44 PLAN  |  Side: {plan['side']} (score={plan['score']})  |  Buildings: {len(encoder.buildings)}"
    line2 = f"Guardian: {g_txt}{fs_txt}"
    cv2.putText(vis, line1, (22, 66), font, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(vis, line2, (22, 96), font, 0.58, (255, 110, 240) if guardian else (190, 190, 190), 2, cv2.LINE_AA)
    return vis


class AnnotatedBattleRecorder:
    """Write a 1200x960 MP4 with opening YOLO summary, live tracking overlays, and decision callouts."""

    def __init__(self, output_path: Path, fps: float = 10.0):
        self.output_path = Path(output_path)
        self.fps = fps
        self.writer = cv2.VideoWriter(
            str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, VIEWPORT
        )
        self.recent_callouts: list[dict] = []  # {"until": t, "text": str, "px": (x,y)|None, "color": bgr}

    def write_still(self, image_bgr: np.ndarray, seconds: float = 2.5):
        if not self.writer.isOpened():
            return
        frame = image_bgr
        if frame.shape[:2] != VIEWPORT[::-1]:
            frame = cv2.resize(frame, VIEWPORT, interpolation=cv2.INTER_AREA)
        for _ in range(max(1, round(seconds * self.fps))):
            self.writer.write(frame)

    def write_banner_frame(self, image_bgr: np.ndarray, headline: str, subline: str = "", target_px=None, seconds: float = 0.1):
        """Write a single frame (or short hold) with a clean phase/decision banner (used during funnel/opening)."""
        if not self.writer.isOpened():
            return
        vis = image_bgr.copy()
        if vis.shape[:2] != VIEWPORT[::-1]:
            vis = cv2.resize(vis, VIEWPORT, interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        box_h = 104 if subline else 72
        cv2.rectangle(vis, (12, 38), (880, box_h), (20, 20, 20), -1)
        cv2.putText(vis, headline, (22, 64), font, 0.60, (245, 245, 245), 2, cv2.LINE_AA)
        if subline:
            cv2.putText(vis, subline, (22, 94), font, 0.54, (70, 230, 255), 2, cv2.LINE_AA)
        if target_px is not None:
            px, py = int(round(target_px[0])), int(round(target_px[1]))
            cv2.circle(vis, (px, py), 24, (15, 15, 15), 5, cv2.LINE_AA)
            cv2.circle(vis, (px, py), 24, (70, 230, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, (px, py), (70, 230, 255), cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
        for _ in range(max(1, round(seconds * self.fps))):
            self.writer.write(vis)

    def add_decision(self, now: float, action, target_px=None, hold_s: float = 3.8):
        slot_name = SLOT_LABELS.get(action.slot, f"SLOT {action.slot}")
        grid_str = f"grid={list(action.grid)}" if action.grid is not None else "ability"
        text = f"DECISION +{now:.1f}s: {slot_name} ({grid_str}) | {action.reason}"
        color = (0, 235, 255) if action.slot == 11 else (
            (255, 100, 255) if action.slot == 13 else (
                (255, 220, 80) if action.slot == 12 else (90, 255, 140)
            )
        )
        self.recent_callouts.append({
            "until": now + hold_s,
            "text": text,
            "px": tuple(int(round(v)) for v in target_px) if target_px is not None else None,
            "color": color,
        })

    def write_step(self, frame_bgr: np.ndarray, now: float, phase: str, groups: list, encoder, transform=None, active_spells=None, live_guardian_tile=None):
        if not self.writer.isOpened():
            return
        vis = frame_bgr.copy()
        if vis.shape[:2] != VIEWPORT[::-1]:
            vis = cv2.resize(vis, VIEWPORT, interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw live tracked Guardian (or shrine anchor) only while Guardian is alive
        if encoder.guardian is not None and encoder.guardian.alive:
            gt = live_guardian_tile if live_guardian_tile is not None else encoder.guardian.tile
            gp = tuple(np.rint(encoder.tile_to_viewport(gt, transform)).astype(int))
            cv2.circle(vis, gp, 22, (15, 15, 15), 4, cv2.LINE_AA)
            cv2.circle(vis, gp, 22, (255, 60, 230), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, gp, (255, 60, 230), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
            cv2.putText(vis, encoder.guardian.name, (gp[0] + 12, gp[1] - 8), font, 0.45, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(vis, encoder.guardian.name, (gp[0] + 12, gp[1] - 8), font, 0.45, (255, 110, 240), 1, cv2.LINE_AA)

        # 2. Draw active spells
        for sp in (active_spells or []):
            if now >= sp["end"]:
                continue
            ring = []
            for ang in np.linspace(0, 2 * math.pi, 36, endpoint=False):
                edge = np.asarray(sp["tile"]) + sp["radius"] * np.array([math.cos(ang), math.sin(ang)])
                ring.append(encoder.tile_to_viewport(edge, transform))
            col = (40, 235, 255) if sp["slot"] == 11 else (255, 210, 60)
            cv2.polylines(vis, [np.int32(ring)], True, col, 2, cv2.LINE_AA)

        # 3. Draw Warden aura + Main Army track
        src_label = "none"
        bar_cnt = 0
        for g in (groups or []):
            if g.get("warden") is not None:
                wc = tuple(np.rint(g["warden"]).astype(int))
                cv2.ellipse(vis, wc, (140, 106), 0, 0, 360, (255, 220, 100), 2, cv2.LINE_AA)
                cv2.drawMarker(vis, wc, (255, 220, 100), cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
            src_label = g["source"]
            bar_cnt = len(g.get("bars", []))
            col = TRACK_COLORS.get(src_label, (255, 255, 255))
            ac = tuple(np.rint(g["center"]).astype(int))
            rad = max(16, round(g.get("radius", 40)))
            cv2.circle(vis, ac, rad, (15, 15, 15), 5, cv2.LINE_AA)
            cv2.circle(vis, ac, rad, col, 2, cv2.LINE_AA)
            cv2.drawMarker(vis, ac, col, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            for bar in g.get("bars", []):
                bp = tuple(np.rint(bar["point"]).astype(int))
                bc = (0, 255, 0) if not bar.get("yellow") else ((0, 0, 255) if bar.get("color") == "red" else (0, 255, 255))
                cv2.circle(vis, bp, 4, bc, -1)

        # 4. Draw active decision callouts (target marker + banner)
        self.recent_callouts = [c for c in self.recent_callouts if now <= c["until"]]
        cv2.rectangle(vis, (12, 38), (820, 74), (20, 20, 20), -1)
        g_name = encoder.guardian.name if encoder.guardian is not None else "none"
        hud_line = f"t=+{now:4.1f}s | {phase} | track={src_label} ({bar_cnt} bars) | guardian={g_name}"
        cv2.putText(vis, hud_line, (22, 62), font, 0.56, (245, 245, 245), 2, cv2.LINE_AA)

        for idx, c in enumerate(self.recent_callouts[-3:]):
            y0 = 80 + idx * 32
            cv2.rectangle(vis, (12, y0), (880, y0 + 28), (18, 18, 18), -1)
            cv2.putText(vis, c["text"], (22, y0 + 20), font, 0.55, c["color"], 2, cv2.LINE_AA)
            if c["px"] is not None:
                px, py = c["px"]
                cv2.circle(vis, (px, py), 24, (15, 15, 15), 5, cv2.LINE_AA)
                cv2.circle(vis, (px, py), 24, c["color"], 2, cv2.LINE_AA)
                cv2.drawMarker(vis, (px, py), c["color"], cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)

        self.writer.write(vis)

    def close(self):
        if self.writer.isOpened():
            self.writer.release()
