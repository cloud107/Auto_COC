"""Track the opening Dragon Duke and estimate nearby anti-air fire for Ice Block."""

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

# Coordinates are on the 1200×960 viewport used by the battle tracker.
VIEWPORT = (1200, 960)
FLIGHT_OFFSET = 37
HEAVY_FIRE = {"air_defense", "monolith", "inferno_tower", "firespitter",
              "firecracker_tower", "revenge_tower", "multi_gear_tower",
              "super_wizard_tower", "spiky_tower", "multi_archer_tower"}


def battlefield_frame(image):
    return cv2.cvtColor(cv2.resize(np.asarray(image), VIEWPORT, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2BGR)


class BattleCamera:
    """Register against distributed static scene patches in the opening screenshot."""

    def __init__(self, frame):
        self.reference = self.gray(frame)

    @staticmethod
    def gray(frame):
        return cv2.cvtColor(cv2.resize(frame, (600, 480), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)

    def align(self, frame):
        gray = self.gray(frame)
        anchors, matches = [], []
        for y in (89, 173, 257, 335):
            for x in (76, 226, 374, 524):
                patch = self.reference[y-13:y+14, x-13:x+14]
                if patch.std() < 11:
                    continue
                search = gray[y-34:y+35, x-34:x+35]
                _, score, _, (u, v) = cv2.minMaxLoc(cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED))
                if score >= .80:
                    anchors.append((x, y))
                    matches.append((x+u-21, y+v-21))
        if len(matches) < 5:
            return None, gray
        transform, inliers = cv2.estimateAffinePartial2D(np.float32(anchors), np.float32(matches),
                                                        ransacReprojThreshold=1.65)
        if transform is None or inliers.sum() < max(5, .65 * len(matches)):
            return None, gray
        supported = np.asarray(anchors)[inliers.ravel().astype(bool)]
        if (np.min(np.ptp(supported, axis=0)) < 145
                or abs(transform[0, 0]-1) > .06 or abs(transform[0, 1]) > .015):
            return None, gray
        transform[:, 2] *= 2
        return transform, gray


@dataclass
class HeroObservation:
    badge: np.ndarray
    position: np.ndarray
    health_width: int


def hero_health(green, mid, starts_at):
    strip = green[max(0, mid-3):mid+3, starts_at:starts_at+37]
    columns = np.flatnonzero(strip.sum(axis=0) >= 2)
    if len(columns) < 3 or columns[0] > 4:
        return None
    # A small visual effect can momentarily darken the middle of the bar.
    limit = np.flatnonzero(np.diff(columns) > 6)
    last = columns[limit[0]] if len(limit) else columns[-1]
    health = int(last - columns[0] + 1)
    return health if health >= 3 else None


def _hero_bar_mask(hue, saturation, value):
    """Match green, yellow, and orange/red friendly hero health bar fills (excluding grass and enemy purple bars)."""
    green = (hue >= 44) & (hue <= 72) & (saturation >= 135) & (value >= 135)
    yellow = (hue >= 19) & (hue <= 36) & (saturation >= 145) & (value >= 160)
    # Keep red strictly hue <= 15 so defending heroes' magenta/purple bars (hue ~150..175) never match.
    red = (hue <= 15) & (saturation >= 150) & (value >= 135)
    return green | yellow | red


def duke_candidates(frame, expected, radius, pre_bgr=None):
    """Read purple level badges with an attached horizontal health bar (green/yellow/red)."""
    x, y = map(round, expected)
    left, top = max(x-radius, 0), max(y-radius, 55)
    right, bottom = min(x+radius, 1200), min(y+radius, 810)
    if right-left < 25 or bottom-top < 14:
        return []
    reg = frame[top:bottom, left:right]
    hue, saturation, value = cv2.split(cv2.cvtColor(reg, cv2.COLOR_BGR2HSV))
    purple = (hue >= 125) & (hue <= 178) & (saturation >= 35) & (value >= 58)
    bar_fill = _hero_bar_mask(hue, saturation, value)
    white = (saturation <= 115) & (value >= 165)
    diff_pre = (np.abs(reg.astype(np.int16) - pre_bgr[top:bottom, left:right].astype(np.int16)).mean(axis=2)
                if pre_bgr is not None else None)
    _, _, stats, _ = cv2.connectedComponentsWithStats(purple.astype(np.uint8))
    observations = []
    for u, v, w, h, area in stats[1:]:
        if not (8 <= w <= 21 and 6 <= h <= 17 and area >= 24
                and np.count_nonzero(white[v:v+h, u:u+w]) >= 10):
            continue
        if diff_pre is not None and diff_pre[v:v+h, u:u+w].mean() < 16.0:
            continue
        mid = v+h//2
        starts_at = u+w-1
        # Reject defending enemy heroes / Guardians whose HP bar to the right of the badge is magenta/purple
        enemy_bar_h = hue[max(0, mid-3):mid+3, starts_at:starts_at+28]
        enemy_bar_s = saturation[max(0, mid-3):mid+3, starts_at:starts_at+28]
        enemy_bar_v = value[max(0, mid-3):mid+3, starts_at:starts_at+28]
        if int(np.count_nonzero((enemy_bar_h >= 132) & (enemy_bar_h <= 168) & (enemy_bar_s >= 100) & (enemy_bar_v >= 90))) >= 8:
            continue
        health = hero_health(bar_fill, mid, starts_at)
        if health is None:
            continue
        badge = np.array([left+u+w/2, top+mid])
        position = np.array([left+starts_at+14, top+mid+FLIGHT_OFFSET])
        observations.append(HeroObservation(badge, position, health))
    return observations


class DragonTracker:
    def __init__(self, seed, pre_bgr=None):
        self.seed = np.asarray(seed, dtype=float)
        self.pre_bgr = pre_bgr
        self.last_badge = None
        self.last_seen = None
        self.consecutive = 0
        self.badge_template = None

    def match_badge(self, frame, expected, radius):
        if self.badge_template is None:
            return None
        x, y = map(round, expected)
        left, right = max(x-radius, 0), min(x+radius, VIEWPORT[0]-40)
        top, bottom = max(y-radius, 55), min(y+radius, 810)
        if right-left < 16 or bottom-top < 12:
            return None
        region = frame[top:bottom, left:right]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(gray, self.badge_template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (u, v) = cv2.minMaxLoc(scores)
        if best < .78:
            return None
        badge_x, badge_y = left+u+7, top+v+5
        starts_at = badge_x+7
        bar = frame[badge_y-3:badge_y+3, starts_at:starts_at+37]
        hue, saturation, value = cv2.split(cv2.cvtColor(bar, cv2.COLOR_BGR2HSV))
        bar_fill = _hero_bar_mask(hue, saturation, value)
        health = hero_health(bar_fill, 3, 0)
        if health is None:
            return None
        return HeroObservation(np.array([badge_x, badge_y]),
                               np.array([starts_at+14, badge_y+FLIGHT_OFFSET]), health)

    def _badge_similarity(self, frame, badge_xy) -> float:
        if self.badge_template is None:
            return 1.0
        bx, by = int(round(badge_xy[0])), int(round(badge_xy[1]))
        th, tw = self.badge_template.shape
        x1, y1 = max(0, bx - tw // 2 - 3), max(0, by - th // 2 - 3)
        x2, y2 = min(frame.shape[1], bx + tw // 2 + 4), min(frame.shape[0], by + th // 2 + 4)
        if x2 - x1 < tw or y2 - y1 < th:
            return 0.0
        patch_g = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(cv2.matchTemplate(patch_g, self.badge_template, cv2.TM_CCOEFF_NORMED).max())

    def observe(self, frame, transform, now, hud_hp_frac=None):
        initial = self.last_badge is None
        expected = self.seed - (21, FLIGHT_OFFSET) if initial else self.last_badge
        gap = 0.0 if initial else max(0.0, now - self.last_seen)
        radius = 85 if initial else min(130, round(35 + 32 * gap))
        current_expected = transform @ np.r_[expected, 1]
        min_hp_w = max(6, int(34.0 * hud_hp_frac * 0.45)) if (hud_hp_frac is not None and hud_hp_frac >= 0.35) else 3
        # Prefer Dragon's own badge template when already learned so nearby defending heroes/flames cannot hijack it.
        measurement = None
        if self.badge_template is not None:
            cand = self.match_badge(frame, current_expected, min(radius, 95))
            if cand is not None and cand.health_width >= min_hp_w:
                measurement = cand
        if measurement is None:
            candidates = duke_candidates(frame, current_expected, radius, pre_bgr=self.pre_bgr)
            filtered = [
                hit for hit in candidates
                if hit.health_width >= min_hp_w
                and (self.badge_template is None or self._badge_similarity(frame, hit.badge) >= 0.58)
            ]
            ranked = sorted(filtered, key=lambda hit: np.linalg.norm(hit.badge-current_expected))
            measurement = ranked[0] if ranked else None
            if len(ranked) > 1 and (initial or np.linalg.norm(ranked[1].badge-current_expected)
                                    - np.linalg.norm(measurement.badge-current_expected) < 15):
                self.consecutive = 0
                return None
        if measurement is None:
            return None
        if self.badge_template is None and measurement.health_width >= 20:
            x, y = map(round, measurement.badge)
            self.badge_template = cv2.cvtColor(frame[y-5:y+6, x-7:x+8], cv2.COLOR_BGR2GRAY)
        inverse = cv2.invertAffineTransform(transform)
        self.last_badge = inverse @ np.r_[measurement.badge, 1]
        self.consecutive = self.consecutive + 1 if not initial and gap <= 1.6 else 1
        self.last_seen = now
        if self.consecutive >= 2 or (not initial and np.linalg.norm(measurement.badge - current_expected) <= 65):
            return measurement
        return None


class DefenseFire:
    """Static YOLO geometry with per-frame visibility checks for in-range towers."""

    def __init__(self, layout, profiles, grid, gray):
        self.grid = grid
        self.towers = []
        self.last_visible = {}
        width = layout["viewport"]["width"]
        height = layout["viewport"]["height"]
        for detection in layout["detections"]:
            name = detection["class"]
            profile = profiles.get(name, {})
            if profile.get("kind") not in {"direct", "directional"}:
                continue
            det_mode = detection.get("mode")
            if det_mode and isinstance(profile.get("modes"), dict) and det_mode in profile["modes"]:
                modes = [profile["modes"][det_mode]]
            else:
                modes = list(profile.get("modes", {"default": profile}).values())
            air_modes = [mode for mode in modes if "air" in mode["targets"]]
            if not air_modes:
                continue
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            x = (x1+x2)/2 * VIEWPORT[0]/width
            bottom = y2 * VIEWPORT[1]/height
            # Estimate the center of the building's footprint from the box lower edge.
            anchor = (x, bottom-7)
            tile = self.tile(anchor)
            patch_x = round(x/2)
            patch_y = round((y1 + .56*(y2-y1)) * VIEWPORT[1]/height/2)
            if not (8 <= patch_x < gray.shape[1]-8 and 8 <= patch_y < gray.shape[0]-8):
                continue
            patch = gray[patch_y-7:patch_y+8, patch_x-7:patch_x+8].copy()
            if patch.std() < 8:
                continue
            facing_rad = float(detection["facing_rad"]) if detection.get("facing_rad") is not None else None
            cone_deg = profile.get("cone_angle_deg")
            cone_half = math.radians(float(cone_deg) * 0.5) if (cone_deg is not None and facing_rad is not None) else None
            known_dir = (len(air_modes) == len(modes)) and (profile["kind"] == "direct" or facing_rad is not None)
            self.towers.append((name, tile, (patch_x, patch_y), patch, air_modes, known_dir, facing_rad, cone_half))

    def tile(self, point):
        return self.grid.pixel_to_grid(point[0]*1000/VIEWPORT[0], point[1]*800/VIEWPORT[1])

    def estimate(self, position, transform, gray, now):
        reference = cv2.invertAffineTransform(transform) @ np.r_[position, 1]
        tile = self.tile(reference)
        confirmed, possible = [], []
        for index, (name, center, patch_point, patch, modes, known_direction, facing_rad, cone_half) in enumerate(self.towers):
            distance = math.dist(tile, center)
            reaches = [mode.get("min_range", 0)+.65 <= distance <= mode["range"]-.65
                       for mode in modes]
            if not any(reaches):
                continue
            if facing_rad is not None and cone_half is not None:
                ang = abs((math.atan2(tile[1] - center[1], tile[0] - center[0]) - facing_rad + math.pi) % (2.0 * math.pi) - math.pi)
                if ang > cone_half:
                    continue
            x, y = np.rint((transform @ np.r_[np.asarray(patch_point)*2, 1])/2).astype(int)
            if not (11 <= x < gray.shape[1]-11 and 11 <= y < gray.shape[0]-11):
                continue
            found = cv2.matchTemplate(gray[y-10:y+11, x-10:x+11], patch, cv2.TM_CCOEFF_NORMED).max()
            if found >= .67:
                self.last_visible[index] = now
            elif now - self.last_visible.get(index, -math.inf) > .8:
                continue
            entry = {"class": name, "distance_tiles": round(distance, 2), "tile": list(center)}
            if known_direction and all(reaches):
                confirmed.append(entry)
            else:
                possible.append(entry)
        return confirmed, possible


class IceBlockPolicy:
    """Require sustained plausible fire and a measured health drop before each cast."""

    def __init__(self, cooldown=7.5):
        self.cooldown = cooldown
        self.retry_after = 0.0
        self.attempts = 0
        self.pressure_at = None
        self.health = deque()
        self.hud_health = deque()

    def evaluate(self, now, health, defenses, hud_hp_frac=None):
        score = sum(2 if item["class"] in HEAVY_FIRE else 1 for item in defenses)
        if self.attempts >= 2 or now < self.retry_after:
            return False, score
        if hud_hp_frac is not None and hud_hp_frac > 0.0:
            self.hud_health.append((now, float(hud_hp_frac)))
            while self.hud_health and now - self.hud_health[0][0] > 1.8:
                self.hud_health.popleft()
        # Real HUD HP rapid-drop / low-HP trigger: saves Dragon Duke even when melted by defending heroes, CC, or Teslas!
        if len(self.hud_health) >= 2:
            peak_hud = max(v for _, v in self.hud_health)
            cur_hud = max(self.hud_health[-1][1], self.hud_health[-2][1])
            if (peak_hud - cur_hud >= 0.18 and cur_hud <= 0.72) or (cur_hud <= 0.50 and peak_hud >= 0.55):
                return True, max(score, 2)
        # Never waste an Ice Block when NO confirmed anti-air defense is in range AND HUD HP isn't dropping!
        if len(defenses) == 0 or score <= 0:
            self.pressure_at = None
            return False, score
        if health is None:
            if self.health and now-self.health[-1][0] > 1.6:
                self.health.clear()
                self.pressure_at = None
            return False, score
        self.health.append((now, health))
        while self.health and now-self.health[0][0] > 2.2:
            self.health.popleft()
        # Cross-check with HUD HP bar so a single occluded battlefield bar frame at full HP never misfires.
        hud_low = (hud_hp_frac is not None) and (0.0 < hud_hp_frac <= 0.62)
        hud_full = (hud_hp_frac is not None) and (hud_hp_frac >= 0.86)
        two_low_bars = len(self.health) >= 2 and (now - self.health[-2][0] <= 1.2) and max(health, self.health[-2][1]) <= 22
        if (hud_low or (two_low_bars and not hud_full)) and len(defenses) >= 1 and score >= 1:
            return True, score
        if len(defenses) < 2 or score < 3:
            self.pressure_at = None
            return False, score
        if self.pressure_at is None:
            self.pressure_at = now
        if len(self.health) < 3 or now-self.health[-2][0] > 1.6:
            return False, score
        peak = max(value for _, value in self.health)
        required_drop = 2 if len(defenses) >= 3 and score >= 4 else 3
        sustained_drop = peak - max(health, self.health[-2][1]) >= required_drop
        measured_decline = (len(self.health) >= 4 and 4 <= peak-health <= 8
                            and max(self.health[-2][1], self.health[-3][1]) < peak)
        return now - self.pressure_at >= .25 and (sustained_drop or measured_decline), score

    def record_attempt(self, now):
        self.attempts += 1
        self.retry_after = now + self.cooldown
        self.pressure_at = None
        self.health.clear()
        self.hud_health.clear()
