"""Encode battlefield perception into an 8x44x44 spatial tensor and global state vector."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from dragon_ice_block import VIEWPORT
from export_yolo_layout import intersection_over_union
from grid_projection import GRID_SIZE
from outer_lawn_deployment import outer_lawn_cells

SPATIAL_CHANNELS = (
    "town_hall",        # Ch 0: Town Hall location heat
    "core_defenses",    # Ch 1: Core/heavy defense footprint & range threat field
    "normal_defenses",  # Ch 2: Standard defense density
    "non_defense",      # Ch 3: Non-defense / resource buildings (reserved until YOLO covers them)
    "main_army",        # Ch 4: Main army cluster position & density
    "heroes",           # Ch 5: Friendly hero positions (Warden / Dragon / etc.)
    "active_spells",    # Ch 6: Active spell coverage weighted by remaining duration
    "deployable_mask",  # Ch 7: Legal outer-lawn deployment mask (1=deployable, 0=restricted)
)

GUARDIAN_CLASSES = {
    "guardian_smasher",
    "guardian_longshot",
    "guardian_log_thrower",
}

CORE_DEFENSE_CLASSES = {
    "scattershot",
    "inferno_tower",
    "monolith",
    "firespitter",
    "revenge_tower",
    "spiky_tower",
    "firecracker_tower",
    "super_wizard_tower",
    "bomb_tower",
    "mortar",
    "spell_tower",
    *GUARDIAN_CLASSES,
}

SPLASH_DEFENSE_CLASSES = {
    "firespitter",
    "spiky_tower",
    "firecracker_tower",
    "super_wizard_tower",
    "inferno_tower",
}

NUM_SLOTS = 16


def _defense_range(profile: dict) -> tuple[float, float]:
    """Return (outer_range, min_range) in tiles for a defense profile."""
    if "range" in profile:
        return float(profile["range"]), float(profile.get("min_range", 0.0))
    if "modes" in profile:
        ranges = [
            float(m.get("range", m.get("trigger_range", 0.0)))
            for m in profile["modes"].values()
        ]
        return (max(ranges) if ranges else 0.0), 0.0
    if "trigger_range" in profile:
        return float(profile["trigger_range"]), 0.0
    if "wake_range" in profile:
        return float(profile["wake_range"]), 0.0
    return 0.0, 0.0


@dataclass
class GridBuilding:
    id: int
    name: str
    tile: np.ndarray  # (gx, gy) in [0, 44)
    is_town_hall: bool
    is_core: bool
    is_splash: bool
    outer_range: float
    min_range: float
    patches: list
    alive: bool = True
    last_seen: float = 0.0
    facing_rad: float | None = None
    cone_half_angle_rad: float | None = None
    mode: str | None = None
    is_non_defense: bool = False

    def covers_tile(self, tile, max_range: float | None = None, margin: float = 0.0) -> bool:
        """Return True if this building can attack `tile` (respecting directional cone if known)."""
        dx = float(tile[0] - self.tile[0])
        dy = float(tile[1] - self.tile[1])
        dist = math.hypot(dx, dy)
        r_max = self.outer_range if max_range is None else max_range
        if dist < self.min_range + margin or dist > r_max:
            return False
        if self.facing_rad is not None and self.cone_half_angle_rad is not None:
            ang = abs((math.atan2(dy, dx) - self.facing_rad + math.pi) % (2.0 * math.pi) - math.pi)
            if ang > self.cone_half_angle_rad:
                return False
        return True


def detect_firespitter_orientations(
    pre_rgb: np.ndarray,
    cone_rgb: np.ndarray,
    layout: dict,
    grid,
    buildings: list[GridBuilding],
) -> list[dict]:
    """Infer 150-degree Firespitter facing angles from a single pre vs tapped-cone frame diff.

    Tapping one Firespitter during pre-deployment displays the bright white 150° range cone
    (outer arc at r≈16 tiles + two radial boundary rays at facing ±75°) for ALL Firespitters
    simultaneously. We score all 8 45°-step directions in tile space using the outer arc and
    radial rays, which gives strong separation even when two Firespitter cones overlap.
    """
    if pre_rgb.shape != cone_rgb.shape:
        return []
    h, w = pre_rgb.shape[:2]
    # White overlay brightens all 3 RGB channels simultaneously.
    wdiff = np.maximum(0.0, np.min(cone_rgb.astype(np.float32) - pre_rgb.astype(np.float32), axis=2))

    def sample_max3(tx: float, ty: float) -> float:
        px, py = grid.grid_to_pixel(tx, ty)
        ix, iy = int(round(px)), int(round(py))
        if 1 <= ix < w - 1 and 1 <= iy < h - 1:
            return float(wdiff[iy - 1:iy + 2, ix - 1:ix + 2].max())
        return 0.0

    # Firespitters in Clash of Clans only rotate to the 4 cardinal isometric grid axes.
    candidate_degs = (0, 90, 180, -90)
    arc_offsets = np.linspace(-72.0, 72.0, 37)
    ray_radii = np.linspace(2.5, 16.5, 22)
    interior_offsets = np.linspace(-65.0, 65.0, 15)
    interior_radii = np.linspace(3.5, 13.5, 10)
    results: list[dict] = []

    fs_detections = [
        d for d in layout.get("detections", [])
        if d.get("class") == "firespitter" and d.get("confidence", 0.0) >= 0.45
    ]

    for b in buildings:
        if b.name != "firespitter":
            continue
        tx, ty = float(b.tile[0]), float(b.tile[1])
        half_deg = math.degrees(b.cone_half_angle_rad) if b.cone_half_angle_rad is not None else 75.0
        r_nom = b.outer_range if b.outer_range > 0 else 16.0
        # Wider radial bracket tolerates slight zoom-out variance across runs
        arc_radii = (r_nom - 2.2, r_nom - 1.2, r_nom - 0.3, r_nom + 0.6, r_nom + 1.5)

        dir_scores: list[tuple[float, int, float, float]] = []
        for deg in candidate_degs:
            rad = math.radians(deg)
            arc_vals = [
                max(sample_max3(tx + r * math.cos(rad + math.radians(off)),
                                ty + r * math.sin(rad + math.radians(off)))
                    for r in arc_radii)
                for off in arc_offsets
            ]
            ray_vals = []
            for sign in (-1.0, 1.0):
                for r in ray_radii:
                    best_a = max(
                        sample_max3(tx + r * math.cos(rad + math.radians(sign * half_deg + da)),
                                    ty + r * math.sin(rad + math.radians(sign * half_deg + da)))
                        for da in (-4.0, 0.0, 4.0)
                    )
                    ray_vals.append(best_a)
            int_vals = [
                sample_max3(tx + r * math.cos(rad + math.radians(off)),
                            ty + r * math.sin(rad + math.radians(off)))
                for off in interior_offsets
                for r in interior_radii
            ]
            # Clip per-sample wdiff to 35.0 before averaging so a single moving defending hero or
            # floating building label (+200 spike at 1-2 points) cannot overpower the consistent white cone lines.
            arc_mean = float(np.mean(np.minimum(arc_vals, 35.0)))
            ray_mean = float(np.mean(np.minimum(ray_vals, 35.0)))
            int_mean = float(np.mean(np.minimum(int_vals, 25.0)))
            total = arc_mean + 2.0 * ray_mean + 0.5 * int_mean
            dir_scores.append((total, deg, arc_mean, ray_mean))

        dir_scores.sort(key=lambda item: item[0], reverse=True)
        best_total, best_deg, best_arc, best_ray = dir_scores[0]
        med_total = float(np.median([s[0] for s in dir_scores]))
        detected = best_total >= 12.0 and best_total >= 1.25 * max(med_total, 1.0)
        if detected:
            b.facing_rad = math.radians(best_deg)
            # Propagate back to matching detection in layout["detections"] for persistence/replay
            vw_src = layout["viewport"]["width"]
            vh_src = layout["viewport"]["height"]
            best_det = None
            best_dist = 1e9
            for d in fs_detections:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                dx_px = (x1 + x2) * 0.5 * w / vw_src
                dy_px = (y2 * h / vh_src) - 8.0 * h / 960.0
                dtx, dty = grid.pixel_to_grid(dx_px, dy_px)
                dist = math.hypot(dtx - tx, dty - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_det = d
            if best_det is not None and best_dist <= 3.0:
                best_det["facing_deg"] = best_deg
                best_det["facing_rad"] = round(b.facing_rad, 4)

        results.append({
            "id": b.id,
            "tile": [round(tx, 2), round(ty, 2)],
            "detected": detected,
            "facing_deg": best_deg if detected else None,
            "score": round(best_total, 1),
            "arc_mean": round(best_arc, 1),
            "ray_mean": round(best_ray, 1),
        })
    return results


def detect_xbow_modes(
    pre_rgb: np.ndarray,
    ring_rgb: np.ndarray,
    layout: dict,
    grid,
    buildings: list[GridBuilding],
) -> list[dict]:
    """Infer X-Bow target modes ('ground' 14-tile vs 'ground_and_air' 11.5-tile) from ONE pre-deployment tap.

    Tapping any X-Bow displays bright white circular range rings for ALL X-Bows simultaneously.
    We compare the white-ring diff energy in the air+ground radial band (~10.0-11.6 tiles)
    against the ground-only radial band (~12.2-14.2 tiles) around each X-Bow.
    """
    if pre_rgb.shape != ring_rgb.shape:
        return []
    h, w = pre_rgb.shape[:2]
    wdiff = np.maximum(0.0, np.min(ring_rgb.astype(np.float32) - pre_rgb.astype(np.float32), axis=2))

    def sample_max3(tx: float, ty: float) -> float:
        px, py = grid.grid_to_pixel(tx, ty)
        ix, iy = int(round(px)), int(round(py))
        if 1 <= ix < w - 1 and 1 <= iy < h - 1:
            return float(wdiff[iy - 1:iy + 2, ix - 1:ix + 2].max())
        return 0.0

    angles = np.linspace(-math.pi, math.pi, 72, endpoint=False)
    air_radii = (10.0, 10.4, 10.8, 11.2, 11.6)
    ground_radii = (12.2, 12.6, 13.0, 13.4, 13.8, 14.2)
    results: list[dict] = []

    xb_detections = [
        d for d in layout.get("detections", [])
        if d.get("class") == "x_bow" and d.get("confidence", 0.0) >= 0.45
    ]

    for b in buildings:
        if b.name != "x_bow":
            continue
        tx, ty = float(b.tile[0]), float(b.tile[1])
        air_vals = [
            min(35.0, max(sample_max3(tx + r * math.cos(a), ty + r * math.sin(a)) for r in air_radii))
            for a in angles
        ]
        ground_vals = [
            min(35.0, max(sample_max3(tx + r * math.cos(a), ty + r * math.sin(a)) for r in ground_radii))
            for a in angles
        ]
        air_score = float(np.mean(air_vals))
        ground_score = float(np.mean(ground_vals))
        best_score = max(air_score, ground_score)
        detected = best_score >= 6.0 and abs(ground_score - air_score) >= 2.0
        mode = ("ground" if ground_score > air_score else "ground_and_air") if detected else None

        if detected and mode is not None:
            b.mode = mode
            b.outer_range = 14.0 if mode == "ground" else 11.5
            vw_src = layout["viewport"]["width"]
            vh_src = layout["viewport"]["height"]
            best_det = None
            best_dist = 1e9
            for d in xb_detections:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                dx_px = (x1 + x2) * 0.5 * w / vw_src
                dy_px = (y2 * h / vh_src) - 8.0 * h / 960.0
                dtx, dty = grid.pixel_to_grid(dx_px, dy_px)
                dist = math.hypot(dtx - tx, dty - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_det = d
            if best_det is not None and best_dist <= 3.0:
                best_det["mode"] = mode

        results.append({
            "id": b.id,
            "tile": [round(tx, 2), round(ty, 2)],
            "detected": detected,
            "mode": mode,
            "ground_score": round(ground_score, 1),
            "air_score": round(air_score, 1),
        })
    return results


class BattlefieldGridEncoder:
    """Maintain alive defense map and encode 10Hz perception into (S_spatial, S_global)."""

    def __init__(self, layout, profiles, grid, pre_image, reference_gray):
        self.grid = grid
        self.buildings: list[GridBuilding] = []
        self._init_buildings(layout, profiles, reference_gray)

        # Ch 7: Legal outer-lawn deployment mask from normalized 1000x800 pre-deployment screenshot
        self.deployable_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        norm_pre = pre_image.convert("RGB")
        if norm_pre.size != (1000, 800):
            norm_pre = norm_pre.resize((1000, 800))
        cells, self.lawn_bounds = outer_lawn_cells(norm_pre, grid, padding=2)
        for col, row in cells:
            c = max(0, min(GRID_SIZE - 1, int(col)))
            r = max(0, min(GRID_SIZE - 1, int(row)))
            self.deployable_mask[r, c] = 1.0

        # Zero-label non-defense building detection (Ch 3) via grass subtraction + 2x2 wall-erasing open
        self.non_defense_buildings: list[GridBuilding] = []
        self._init_non_defense_buildings(norm_pre, reference_gray)

        # Precompute 44x44 tile center coordinates for fast vectorized splatting
        yy, xx = np.mgrid[:GRID_SIZE, :GRID_SIZE]
        self._gx = xx.astype(np.float32) + 0.5
        self._gy = yy.astype(np.float32) + 0.5

    def _init_non_defense_buildings(self, norm_pre, reference_gray):
        """Detect non-defense buildings inside the red border with zero manual labels (strips grass, walls, & YOLO boxes)."""
        if self.lawn_bounds is None:
            return
        left, right, top, bottom = self.lawn_bounds
        rgb = np.asarray(norm_pre, dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        grass = (h >= 31) & (h <= 59) & (s >= 85) & (v >= 112)
        gh, gw = grass.shape

        occ = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        c_lo, c_hi = max(0, int(math.ceil(left + 0.5))), min(GRID_SIZE, int(math.floor(right - 0.5)))
        r_lo, r_hi = max(0, int(math.ceil(top + 0.5))), min(GRID_SIZE, int(math.floor(bottom - 0.5)))
        for row in range(r_lo, r_hi):
            for col in range(c_lo, c_hi):
                px, py = self.grid.grid_to_pixel(col + 0.5, row + 0.5)
                ix, iy = int(round(px)), int(round(py))
                if 6 <= ix < gw - 6 and 70 <= iy < int(gh * 0.84):
                    if grass[iy - 3:iy + 4, ix - 3:ix + 4].mean() < 0.58:
                        occ[row, col] = 1

        # Erase already-detected YOLO defenses, Town Hall, Clan Castle, Guardian footprints
        for b in self.buildings:
            bc, br = int(round(float(b.tile[0]))), int(round(float(b.tile[1])))
            rad = 3 if b.name in ("town_hall", "clan_castle") else 2
            cv2.circle(occ, (bc, br), rad, 0, -1)

        # 2x2 morphological opening in 44x44 tile space erases all 1-tile-thick Walls while keeping 3x3+ buildings!
        opened = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened)

        next_id = 1000
        for lbl in range(1, num_labels):
            x, y, w, bh, area = stats[lbl]
            if area < 4:
                continue
            # Split wide adjacent trash clusters into ~3.5-tile building centers
            if w <= 5 and bh <= 5:
                centers = [(float(centroids[lbl][0]), float(centroids[lbl][1]))]
            else:
                centers = []
                ys_c, xs_c = np.where(labels == lbl)
                for r0 in range(y, y + bh, 3):
                    for c0 in range(x, x + w, 3):
                        sub = opened[r0:min(y + bh, r0 + 3), c0:min(x + w, c0 + 3)]
                        if int(sub.sum()) >= 4:
                            centers.append((float(c0 + 1.5), float(r0 + 1.5)))
            for cx, cy in centers:
                if any(math.hypot(cx - b.tile[0], cy - b.tile[1]) < 2.4 for b in self.buildings):
                    continue
                if any(math.hypot(cx - nb.tile[0], cy - nb.tile[1]) < 2.6 for nb in self.non_defense_buildings):
                    continue
                vp = self.tile_to_viewport((cx, cy))
                u, v_ref = int(round(vp[0] / 2.0)), int(round(vp[1] / 2.0))
                patches = []
                for dv in (-4, 4):
                    vv = v_ref + dv
                    if 8 <= u < reference_gray.shape[1] - 8 and 8 <= vv < reference_gray.shape[0] - 8:
                        patch = reference_gray[vv - 6:vv + 7, u - 6:u + 7].copy()
                        if patch.std() >= 15.0:
                            patches.append(((u, vv), patch))
                # Reject uniform shaded grass tiles that have no real building texture
                if not patches:
                    continue
                self.non_defense_buildings.append(GridBuilding(
                    id=next_id,
                    name="non_defense",
                    tile=np.array([cx, cy], dtype=np.float32),
                    is_town_hall=False,
                    is_core=False,
                    is_splash=False,
                    outer_range=0.0,
                    min_range=0.0,
                    patches=patches,
                    alive=True,
                    last_seen=0.0,
                    is_non_defense=True,
                ))
                next_id += 1

    def _init_buildings(self, layout, profiles, reference_gray):
        self.guardian: GridBuilding | None = None
        width = layout["viewport"]["width"]
        height = layout["viewport"]["height"]
        detections = sorted(layout["detections"], key=lambda d: d["confidence"], reverse=True)
        seen = []
        for idx, d in enumerate(detections):
            name, box = d["class"], d["bbox_xyxy"]
            # Keep normal buildings at >=0.45; allow unique Guardian detection down to >=0.30.
            if name not in GUARDIAN_CLASSES and d.get("confidence", 1.0) < 0.45:
                continue
            # Only one Guardian exists per base: keep the single highest-confidence Guardian detection.
            if name in GUARDIAN_CLASSES and self.guardian is not None:
                continue
            if any(
                (name == prev["class"] or (name in GUARDIAN_CLASSES and prev["class"] in GUARDIAN_CLASSES))
                and intersection_over_union(box, prev["bbox_xyxy"]) > 0.28
                for prev in seen
            ):
                continue
            seen.append(d)
            x1, y1, x2, y2 = box
            vx = (x1 + x2) * 0.5 * VIEWPORT[0] / width
            vy = y2 * VIEWPORT[1] / height
            tile = self.viewport_to_tile((vx, vy - 8.0))
            patches = []
            u = round(vx / 2)
            for frac in (0.48, 0.80):
                v = round((y1 + frac * (y2 - y1)) * VIEWPORT[1] / height / 2)
                if 8 <= u < reference_gray.shape[1] - 8 and 8 <= v < reference_gray.shape[0] - 8:
                    patch = reference_gray[v - 6:v + 7, u - 6:u + 7].copy()
                    if patch.std() >= 10:
                        patches.append(((u, v), patch))
            profile_key = "multi_gear_tower" if name == "multi_gear_tower_variant" else name
            profile = profiles.get(profile_key, {})
            outer_r, min_r = _defense_range(profile)
            det_mode = d.get("mode")
            if det_mode and isinstance(profile.get("modes"), dict) and det_mode in profile["modes"]:
                m_cfg = profile["modes"][det_mode]
                outer_r = float(m_cfg.get("range", m_cfg.get("trigger_range", outer_r)))
            cone_deg = profile.get("cone_angle_deg")
            cone_half = math.radians(float(cone_deg) * 0.5) if cone_deg is not None else None
            facing_rad = float(d["facing_rad"]) if d.get("facing_rad") is not None else None
            b = GridBuilding(
                id=idx,
                name=name,
                tile=tile,
                is_town_hall=(name == "town_hall"),
                is_core=(name in CORE_DEFENSE_CLASSES),
                is_splash=(name in SPLASH_DEFENSE_CLASSES),
                outer_range=outer_r,
                min_range=min_r,
                patches=patches,
                alive=True,
                last_seen=0.0,
                facing_rad=facing_rad,
                cone_half_angle_rad=cone_half,
                mode=det_mode,
            )
            self.buildings.append(b)
            if name in GUARDIAN_CLASSES:
                self.guardian = b

    def viewport_to_tile(self, point) -> np.ndarray:
        """Convert 1200x960 viewport pixel (x, y) to 44x44 tile coordinates (gx, gy)."""
        gx, gy = self.grid.pixel_to_grid(point[0] * 1000.0 / VIEWPORT[0], point[1] * 800.0 / VIEWPORT[1])
        return np.array([float(gx), float(gy)], dtype=np.float32)

    def tile_to_viewport(self, tile, transform=None) -> np.ndarray:
        """Convert 44x44 tile coordinates (gx, gy) back to 1200x960 viewport pixel (x, y)."""
        x, y = self.grid.grid_to_pixel(float(tile[0]), float(tile[1]))
        pt = np.array([x * VIEWPORT[0] / 1000.0, y * VIEWPORT[1] / 800.0, 1.0], dtype=float)
        if transform is not None:
            return transform @ pt
        return pt[:2]

    def update_alive_buildings(self, now: float, transform, gray) -> None:
        """Update alive flags for defenses AND zero-label non-defense buildings using NCC against pre.png."""
        if transform is None or gray is None:
            return
        for b in (*self.buildings, *self.non_defense_buildings):
            if not b.patches:
                continue
            matched = False
            for patch_center, patch in b.patches:
                x, y = np.rint((transform @ np.r_[np.asarray(patch_center) * 2.0, 1.0]) / 2.0).astype(int)
                if not (12 <= x < gray.shape[1] - 12 and 12 <= y < gray.shape[0] - 12):
                    continue
                sim = cv2.matchTemplate(
                    gray[y - 11:y + 12, x - 11:x + 12], patch, cv2.TM_CCOEFF_NORMED
                ).max()
                if sim >= 0.72:
                    matched = True
                    b.last_seen = now
                    break
            b.alive = matched or ((now - b.last_seen) <= 1.5)

    def _splat_disk(self, channel: np.ndarray, center, radius: float, value: float, add: bool = True):
        dist_sq = (self._gx - center[0]) ** 2 + (self._gy - center[1]) ** 2
        mask = dist_sq <= (radius ** 2)
        if add:
            channel[mask] += value
        else:
            channel[mask] = np.maximum(channel[mask], value)

    def _splat_gaussian(self, channel: np.ndarray, center, sigma: float, weight: float = 1.0):
        dist_sq = (self._gx - center[0]) ** 2 + (self._gy - center[1]) ** 2
        channel += weight * np.exp(-0.5 * dist_sq / (sigma ** 2))

    def encode(
        self,
        now: float,
        *,
        transform=None,
        gray=None,
        army_groups: list[dict] | None = None,
        extra_heroes: list[tuple[float, float]] | None = None,
        active_spells: list[dict] | None = None,
        card_counts: dict[int, int] | None = None,
        hero_states: list[int] | None = None,
        destruction_pct: float | None = None,
    ) -> dict:
        """Build S_spatial (8, 44, 44) and S_global (D,) for the current 10Hz decision tick."""
        if transform is not None and gray is not None:
            self.update_alive_buildings(now, transform, gray)
        inv = cv2.invertAffineTransform(transform) if transform is not None else None

        spatial = np.zeros((len(SPATIAL_CHANNELS), GRID_SIZE, GRID_SIZE), dtype=np.float32)

        alive_count = 0
        for b in self.buildings:
            if not b.alive:
                continue
            alive_count += 1
            if b.is_town_hall:
                self._splat_gaussian(spatial[0], b.tile, sigma=1.6, weight=1.0)
            elif b.is_core:
                # Footprint peak + range threat field (directional cone if known)
                self._splat_gaussian(spatial[1], b.tile, sigma=1.2, weight=1.0)
                if b.outer_range > 0:
                    dx = self._gx - b.tile[0]
                    dy = self._gy - b.tile[1]
                    dist = np.hypot(dx, dy)
                    if b.facing_rad is not None and b.cone_half_angle_rad is not None:
                        ang = np.abs((np.arctan2(dy, dx) - b.facing_rad + math.pi) % (2.0 * math.pi) - math.pi)
                        in_rng = (dist >= b.min_range) & (dist <= b.outer_range) & (ang <= b.cone_half_angle_rad)
                    else:
                        r_eff = min(b.outer_range, 10.5) if b.cone_half_angle_rad is not None else b.outer_range
                        in_rng = (dist >= b.min_range) & (dist <= r_eff)
                    spatial[1][in_rng] += 0.25
            else:
                self._splat_gaussian(spatial[2], b.tile, sigma=1.2, weight=1.0)

        # Ch 3: Alive zero-label non-defense buildings
        for nb in self.non_defense_buildings:
            if nb.alive:
                self._splat_gaussian(spatial[3], nb.tile, sigma=1.1, weight=1.0)

        # Ch 4: Main army cluster & Ch 5: Heroes
        army_tile = None
        army_vel_tile = np.zeros(2, dtype=np.float32)
        warden_tile = None
        for g in (army_groups or []):
            center_px = np.asarray(g["center"], dtype=float)
            if inv is not None:
                center_px = inv @ np.r_[center_px, 1.0]
            c_tile = self.viewport_to_tile(center_px)
            if army_tile is None:
                army_tile = c_tile
                vel_px = np.asarray(g.get("velocity", [0.0, 0.0]), dtype=float)
                if inv is not None:
                    vel_px = inv[:, :2] @ vel_px
                # Approximate tile velocity via local finite difference
                c_next = self.viewport_to_tile(center_px + vel_px)
                army_vel_tile = c_next - c_tile
            sigma_tiles = max(1.3, float(g.get("radius", 40.0)) / 18.0)
            self._splat_gaussian(spatial[4], c_tile, sigma=sigma_tiles, weight=1.0)
            for bar in g.get("bars", []):
                bp = np.asarray(bar["point"], dtype=float)
                if inv is not None:
                    bp = inv @ np.r_[bp, 1.0]
                bt = self.viewport_to_tile(bp)
                self._splat_gaussian(spatial[4], bt, sigma=0.7, weight=0.4)
            if g.get("warden") is not None:
                wp = np.asarray(g["warden"], dtype=float)
                if inv is not None:
                    wp = inv @ np.r_[wp, 1.0]
                warden_tile = self.viewport_to_tile(wp)
                self._splat_gaussian(spatial[5], warden_tile, sigma=1.1, weight=1.0)

        for hp in (extra_heroes or []):
            pt = np.asarray(hp, dtype=float)
            if inv is not None:
                pt = inv @ np.r_[pt, 1.0]
            self._splat_gaussian(spatial[5], self.viewport_to_tile(pt), sigma=1.1, weight=1.0)

        # Ch 6: Active spell coverage weighted by remaining duration fraction
        for sp in (active_spells or []):
            rem = float(sp["end"] - now)
            if rem <= 0:
                continue
            frac = min(1.0, rem / max(0.1, float(sp["duration"])))
            self._splat_disk(spatial[6], sp["tile"], float(sp["radius"]), frac, add=False)

        # Ch 7: Deployable mask
        spatial[7] = self.deployable_mask

        # S_global: [t/180, destruction_est, 16x card_counts, 4x hero_states] -> 22 floats
        if destruction_pct is None:
            total_b = max(1, len(self.buildings))
            destruction_est = 1.0 - (alive_count / total_b)
        else:
            destruction_est = float(destruction_pct) / 100.0
        counts_vec = [float((card_counts or {}).get(slot, 0)) / 5.0 for slot in range(1, NUM_SLOTS + 1)]
        heroes_vec = [float(h) / 2.0 for h in (hero_states or [0, 0, 0, 0])[:4]]
        global_vec = np.array([min(1.0, now / 180.0), destruction_est, *counts_vec, *heroes_vec], dtype=np.float32)

        return {
            "time": round(now, 3),
            "spatial": spatial,
            "global": global_vec,
            "army_tile": army_tile,
            "army_vel_tile": army_vel_tile,
            "warden_tile": warden_tile,
            "guardian": self.guardian,
            "alive_buildings": [b for b in self.buildings if b.alive],
            "alive_non_defense": [nb for nb in self.non_defense_buildings if nb.alive],
        }


class GuardianTracker:
    """Continuously track the walking Guardian from its pre-battle shrine using its masked level badge."""

    def __init__(self, shrine_px, pre_bgr):
        self.shrine_px = np.asarray(shrine_px, dtype=float)
        self.pre_bgr = pre_bgr.copy()
        self.badge_pos = None
        self.init_badge = None
        self.tmpl = None
        self.tmpl_mask = None
        self.last_seen = 0.0
        self.awake = False
        self.alive = True
        # Foot position sits ~48px directly below the Guardian's level badge (1200x960 viewport)
        self.foot_offset = np.array([10.0, 48.0], dtype=float)
        self._init_from_pre(pre_bgr)

    def _init_from_pre(self, pre_bgr):
        # Tight window right above the Guardian perch (y - 95 .. y - 28; badge sits ~75px above feet)
        l = max(0, int(self.shrine_px[0] - 42))
        t = max(55, int(self.shrine_px[1] - 95))
        r = min(VIEWPORT[0], int(self.shrine_px[0] + 42))
        bt = min(820, int(self.shrine_px[1] - 28))
        if r <= l + 20 or bt <= t + 10:
            return
        reg = pre_bgr[t:bt, l:r]
        h, s, v = cv2.split(cv2.cvtColor(reg, cv2.COLOR_BGR2HSV))
        purple = (h >= 125) & (h <= 170) & (s >= 65) & (v >= 80)
        _, _, stats, _ = cv2.connectedComponentsWithStats(purple.astype(np.uint8))
        bars = []
        for x, y, w, bh, area in stats[1:]:
            if 12 <= w <= 56 and 2 <= bh <= 7 and w >= 2.2 * bh and area >= 0.48 * w * bh:
                bars.append((l + x, t + y, w, bh))
        if not bars:
            return
        exp_x = self.shrine_px[0]
        exp_y = self.shrine_px[1] - 75.0
        bx, by, bw, bh = min(
            bars,
            key=lambda item: math.hypot((item[0] + item[2] / 2.0) - exp_x, (item[1] + item[3] / 2.0) - exp_y),
        )
        cx, cy = int(bx - 6), int(by + bh // 2)
        if not (10 <= cx < VIEWPORT[0] - 10 and 10 <= cy < VIEWPORT[1] - 10):
            return
        # Tight 9x11 inner shield+digit crop with core mask so surrounding walls/buildings never corrupt correlation.
        patch_bgr = pre_bgr[cy - 5:cy + 6, cx - 4:cx + 5]
        self.tmpl = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(self.tmpl)
        mask[1:-1, 1:-1] = 255
        self.tmpl_mask = mask
        self.badge_pos = np.array([float(cx), float(cy)], dtype=float)
        self.init_badge = self.badge_pos.copy()
        # Pedestal offset when perched on shrine vs ground-level walking offset (measured +10px X, +41px Y below [5] badge)
        delta = self.shrine_px - self.init_badge
        self.foot_offset = np.array([float(np.clip(delta[0], 0.0, 32.0)), float(np.clip(delta[1], 45.0, 92.0))], dtype=float)
        self.walk_foot_offset = np.array([10.0, 41.0], dtype=float)

    @staticmethod
    def _has_badge_glyph(patch_bgr) -> bool:
        """Verify a candidate [5] badge patch actually has dark badge background + bright white digit pixels."""
        if patch_bgr is None or patch_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        dark_frac = float(np.count_nonzero(v < 95)) / max(1, v.size)
        white_px = int(np.count_nonzero((v >= 185) & (s <= 75)))
        return dark_frac >= 0.20 and white_px >= 5

    def observe(self, frame_bgr, now: float, transform=None):
        """Leap-proof continuous Guardian tracking across the entire base; returns (feet_px | None, awake)."""
        if self.tmpl is None or self.badge_pos is None or self.init_badge is None:
            return (self.shrine_px.copy(), False) if self.alive else (None, False)
        anchor = (transform @ np.r_[self.init_badge, 1.0]) if transform is not None else self.init_badge
        inv = cv2.invertAffineTransform(transform) if transform is not None else None
        th, tw = self.tmpl.shape

        # Step 0: Check if the [5] badge is STILL sitting right at the shrine (init_badge)!
        # While the Guardian has not moved off its shrine, NEVER let a nearby damaged defense's purple HP bar hijack it!
        ix, iy = int(round(anchor[0])), int(round(anchor[1]))
        sx0, sx1 = max(0, ix - tw // 2 - 5), min(frame_bgr.shape[1], ix + tw // 2 + 6)
        sy0, sy1 = max(0, iy - th // 2 - 5), min(frame_bgr.shape[0], iy + th // 2 + 6)
        if sx1 - sx0 >= tw and sy1 - sy0 >= th:
            shrine_sub = frame_bgr[sy0:sy1, sx0:sx1]
            shrine_gray = cv2.cvtColor(shrine_sub, cv2.COLOR_BGR2GRAY)
            shrine_res = cv2.matchTemplate(shrine_gray, self.tmpl, cv2.TM_CCOEFF_NORMED, mask=self.tmpl_mask)
            shrine_res = np.nan_to_num(shrine_res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            _, shrine_sc, _, (su, sv) = cv2.minMaxLoc(shrine_res)
            if shrine_sc >= 0.78 and self._has_badge_glyph(shrine_sub[sv:sv + th, su:su + tw]):
                cand_scr = np.array([sx0 + su + tw / 2.0, sy0 + sv + th / 2.0])
                self.badge_pos = (inv @ np.r_[cand_scr, 1.0]) if inv is not None else cand_scr
                self.last_seen = now
                self.alive = True
                self.awake = False
                return (self.badge_pos + self.foot_offset), False

        if not self.alive:
            return None, False

        prev_scr = (transform @ np.r_[self.badge_pos, 1.0]) if transform is not None else self.badge_pos
        # Velocity-gated search radius so nearby defending Heroes ([100]/[115]) can NEVER hijack the Guardian!
        dt_gap = max(0.0, now - self.last_seen) if self.last_seen > 0.0 else 0.0
        max_step = 80.0 if not self.awake else min(115.0, 32.0 + 38.0 * dt_gap)
        rad = int(round(max_step)) + 20
        cx_min = min(anchor[0], prev_scr[0]) if not self.awake else prev_scr[0]
        cx_max = max(anchor[0], prev_scr[0]) if not self.awake else prev_scr[0]
        cy_min = min(anchor[1], prev_scr[1]) if not self.awake else prev_scr[1]
        cy_max = max(anchor[1], prev_scr[1]) if not self.awake else prev_scr[1]
        xl, yl = max(0, int(round(cx_min)) - rad), max(55, int(round(cy_min)) - rad)
        xr, yr = min(VIEWPORT[0], int(round(cx_max)) + rad), min(820, int(round(cy_max)) + rad)
        if xr - xl > tw + 32 and yr - yl > th + 4:
            sub_bgr = frame_bgr[yl:yr, xl:xr]
            gray = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, self.tmpl, cv2.TM_CCOEFF_NORMED, mask=self.tmpl_mask)
            res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
            best_cand_ref = None
            best_score = -1e9

            # Pass 1: template match peaks (>= 0.72) strictly within velocity gate + real [5] glyph + purple HP bar
            ys, xs = np.where(res >= 0.72)
            if len(xs) > 0:
                order = np.argsort(res[ys, xs])[::-1][:30]
                for k in order:
                    u, v_i = int(xs[k]), int(ys[k])
                    sc = float(res[v_i, u])
                    cur_patch = sub_bgr[v_i:v_i + th, u:u + tw]
                    if not self._has_badge_glyph(cur_patch):
                        continue
                    cand_scr = np.array([xl + u + tw / 2.0, yl + v_i + th / 2.0])
                    dist_prev = float(np.linalg.norm(cand_scr - prev_scr))
                    if dist_prev > max_step:
                        continue
                    cand_ref = (inv @ np.r_[cand_scr, 1.0]) if inv is not None else cand_scr
                    dist_init = float(np.linalg.norm(cand_ref - self.init_badge))
                    if dist_init >= 8.0:
                        rx, ry = int(round(cand_ref[0] - tw / 2.0)), int(round(cand_ref[1] - th / 2.0))
                        if 0 <= rx and rx + tw <= self.pre_bgr.shape[1] and 0 <= ry and ry + th <= self.pre_bgr.shape[0]:
                            pre_patch = self.pre_bgr[ry:ry + th, rx:rx + tw]
                            if float(np.abs(cur_patch.astype(np.int16) - pre_patch.astype(np.int16)).mean()) < 18.0:
                                continue
                    bx_s, by_s = u + tw, v_i + th // 2
                    strip = sub_bgr[max(0, by_s - 3):by_s + 4, bx_s:min(sub_bgr.shape[1], bx_s + 32)]
                    purple_px = 0
                    if strip.size > 0:
                        sh, ss, sv = cv2.split(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV))
                        purple_px = int(np.count_nonzero((sh >= 125) & (sh <= 170) & (ss >= 65) & (sv >= 80)))
                    if purple_px >= 6 or (sc >= 0.88 and dist_init >= 8.0):
                        combined = sc - 0.0020 * dist_prev
                        if combined > best_score:
                            best_score = combined
                            best_cand_ref = cand_ref

            # Pass 2: purple-HP-bar-first fallback strictly requiring BOTH real [5] glyph AND match >= 0.70
            # (rejects damaged defense buildings that have purple HP bars with NO level badge!)
            if best_cand_ref is None:
                sub_hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
                ph, ps, pv = cv2.split(sub_hsv)
                purple = (ph >= 128) & (ph <= 170) & (ps >= 95) & (pv >= 90)
                _, _, stats, _ = cv2.connectedComponentsWithStats(purple.astype(np.uint8))
                for x_p, y_p, w_p, h_p, area_p in stats[1:]:
                    if not (10 <= w_p <= 56 and 2 <= h_p <= 7 and w_p >= 2.2 * h_p and area_p >= 0.45 * w_p * h_p):
                        continue
                    # Reject wide 2-3 digit defending Hero badges ([60]/[90]/[100]/[115]): check white digits >13px left!
                    far_left = sub_hsv[max(0, y_p - 4):y_p + h_p + 4, max(0, x_p - 22):max(0, x_p - 13)]
                    if far_left.size > 0:
                        white_far = int(np.count_nonzero((far_left[:, :, 1] <= 80) & (far_left[:, :, 2] >= 185)))
                        if white_far >= 8:
                            continue
                    u0 = max(0, x_p - tw - 2)
                    u1 = min(res.shape[1], x_p - tw + 4)
                    v0 = max(0, y_p + h_p // 2 - th // 2 - 3)
                    v1 = min(res.shape[0], y_p + h_p // 2 - th // 2 + 4)
                    if u1 <= u0 or v1 <= v0:
                        continue
                    local_res = res[v0:v1, u0:u1]
                    _, lsc, _, (lu, lv) = cv2.minMaxLoc(local_res)
                    if lsc < 0.70:
                        continue
                    u, v_i = u0 + lu, v0 + lv
                    cur_patch = sub_bgr[v_i:v_i + th, u:u + tw]
                    if not self._has_badge_glyph(cur_patch):
                        continue
                    cand_scr = np.array([xl + u + tw / 2.0, yl + v_i + th / 2.0])
                    dist_prev = float(np.linalg.norm(cand_scr - prev_scr))
                    if dist_prev > max_step:
                        continue
                    cand_ref = (inv @ np.r_[cand_scr, 1.0]) if inv is not None else cand_scr
                    dist_init = float(np.linalg.norm(cand_ref - self.init_badge))
                    if dist_init >= 8.0:
                        rx, ry = int(round(cand_ref[0] - tw / 2.0)), int(round(cand_ref[1] - th / 2.0))
                        if 0 <= rx and rx + tw <= self.pre_bgr.shape[1] and 0 <= ry and ry + th <= self.pre_bgr.shape[0]:
                            pre_patch = self.pre_bgr[ry:ry + th, rx:rx + tw]
                            if float(np.abs(cur_patch.astype(np.int16) - pre_patch.astype(np.int16)).mean()) < 16.0:
                                continue
                    combined = lsc - 0.0022 * dist_prev
                    if combined > best_score:
                        best_score = combined
                        best_cand_ref = cand_ref

            if best_cand_ref is not None:
                step = float(np.linalg.norm(best_cand_ref - self.badge_pos))
                # Snap immediately on opening leap (>25px), smooth gently during walking
                self.badge_pos = best_cand_ref if step > 25.0 else (0.75 * best_cand_ref + 0.25 * self.badge_pos)
                self.last_seen = now
                if float(np.linalg.norm(self.badge_pos - self.init_badge)) >= 12.0:
                    self.awake = True
            elif self.awake and (now - self.last_seen) >= 7.0:
                # Guardian woke up and engaged, then its [5]+purple HP bar vanished for >=7s -> KO'd!
                self.alive = False
                return None, False
        offset = self.walk_foot_offset if self.awake else self.foot_offset
        return (self.badge_pos + offset), self.awake


def detect_enemy_hero_bars(image, pre_bgr, roi):
    """Detect defending enemy hero/Guardian purple health bars (zero-label Guardian detection)."""
    left, top, right, bottom = roi
    if right <= left + 8 or bottom <= top + 8:
        return []
    region = image[top:bottom, left:right]
    pre_reg = pre_bgr[top:bottom, left:right]
    hue, sat, val = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HSV))
    # Defending heroes & Guardian have a distinct magenta/purple HP bar with dark border.
    purple = (hue >= 132) & (hue <= 165) & (sat >= 120) & (val >= 100)
    dark = val < 85
    diff_pre = np.abs(region.astype(np.int16) - pre_reg.astype(np.int16)).mean(axis=2)
    _, _, stats, _ = cv2.connectedComponentsWithStats(purple.astype(np.uint8))
    bars = []
    for x, y, w, h, area in stats[1:]:
        if not (16 <= w <= 56 and 2 <= h <= 6 and w >= 3.2 * h and area >= 0.60 * w * h):
            continue
        if y < 2 or y + h + 3 > region.shape[0]:
            continue
        outline = max(dark[y - 2:y, x:x + w].mean(), dark[y + h:y + h + 3, x:x + w].mean())
        if outline < 0.35 or diff_pre[y:y + h, x:x + w].mean() < 18.0:
            continue
        bars.append((float(left + x + w / 2.0), float(top + y + h + 18.0)))
    return bars


class TrajectoryRecorder:
    """Record 10Hz (S_spatial, S_global, masks, a_slot, a_grid) steps for Stage 2 Behavior Cloning."""

    def __init__(self):
        self.spatials = []
        self.globals = []
        self.slot_masks = []
        self.grid_masks = []
        self.slots = []
        self.grids = []
        self.times = []

    def add(self, state: dict, action) -> None:
        self.times.append(float(state["time"]))
        self.spatials.append(state["spatial"].astype(np.float16))
        self.globals.append(state["global"].astype(np.float32))
        self.slot_masks.append(
            action.slot_mask if action.slot_mask is not None else np.ones(NUM_SLOTS + 1, dtype=bool)
        )
        self.grid_masks.append(
            action.grid_mask if action.grid_mask is not None else np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        )
        self.slots.append(int(action.slot))
        if action.grid is not None:
            self.grids.append((int(action.grid[0]), int(action.grid[1])))
        else:
            self.grids.append((-1, -1))

    def save(self, path) -> int:
        if not self.times:
            return 0
        np.savez_compressed(
            str(path),
            times=np.asarray(self.times, dtype=np.float32),
            S_spatial=np.stack(self.spatials, axis=0),
            S_global=np.stack(self.globals, axis=0),
            slot_mask=np.stack(self.slot_masks, axis=0),
            grid_mask=np.stack(self.grid_masks, axis=0),
            a_slot=np.asarray(self.slots, dtype=np.int16),
            a_grid=np.asarray(self.grids, dtype=np.int16),
        )
        return len(self.times)


def render_ascii(spatial: np.ndarray) -> str:
    """Render the 8x44x44 spatial tensor as a readable 44x44 ASCII map for verification."""
    lines = []
    for row in range(GRID_SIZE):
        chars = []
        for col in range(GRID_SIZE):
            if spatial[0, row, col] >= 0.45:
                ch = "T"  # Town Hall
            elif spatial[4, row, col] >= 0.55:
                ch = "M"  # Main army
            elif spatial[5, row, col] >= 0.55:
                ch = "W"  # Grand Warden / Hero
            elif spatial[6, row, col] >= 0.15:
                ch = "+"  # Active spell
            elif spatial[1, row, col] >= 0.85:
                ch = "C"  # Core defense
            elif spatial[2, row, col] >= 0.55:
                ch = "d"  # Normal defense
            elif spatial[7, row, col] >= 0.5:
                ch = "."  # Legal deployable lawn
            else:
                ch = " "
            chars.append(ch)
        lines.append("".join(chars))
    return "\n".join(lines)
