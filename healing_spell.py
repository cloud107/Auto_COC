"""Estimate where Miner/Hog groups will stop and evaluate healing spell placement."""

import math
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

from deployed_wave_filter import DeployedWaveTracks, opening_prior, split_group_measurements
from dragon_ice_block import VIEWPORT
from export_yolo_layout import intersection_over_union
from health_bar_tracks import HealthBarTracks
from track_friendly_activity import (
    deployment_group, follow_hidden_wave, has_level_badge, health_bars, motion_evidence, nearby_motion,
)

HEAL_RADIUS = 5.0
HEAL_DURATION = 12.0
# Reacquisition distance in the 1200x960 tracking viewport.
MAX_INITIAL_WAVE_REACQUIRE_DISTANCE = 210
SPLASH_DEFENSES = {"firecracker_tower", "scattershot", "super_wizard_tower", "bomb_tower", "mortar"}


def _make_warden_ring_template(rx_small=70.0, ratio=0.757):
    """Build a zero-mean radial Mexican-hat template for the Grand Warden's aura ellipse at 600x480."""
    ry = round(rx_small * ratio)
    pad = 6
    w, h = 2 * (int(round(rx_small)) + pad) + 1, 2 * (ry + pad) + 1
    cx, cy = int(round(rx_small)) + pad, ry + pad
    yy, xx = np.mgrid[:h, :w]
    d = np.sqrt(((xx - cx) / rx_small) ** 2 + ((yy - cy) / ry) ** 2)
    rim = np.exp(-0.5 * ((d - 1.0) / 0.022) ** 2)
    inner_glow = 0.25 * np.exp(-0.5 * ((d - 0.90) / 0.06) ** 2)
    outer_neg = -0.85 * np.exp(-0.5 * ((d - 1.06) / 0.03) ** 2)
    tmpl = rim + inner_glow + outer_neg
    tmpl -= tmpl.mean()
    tmpl /= (np.linalg.norm(tmpl) + 1e-6)
    return tmpl.astype(np.float32)


WARDEN_RING_TEMPLATE = _make_warden_ring_template(70.0)


def detect_troop_health_bars(image, pre_bgr, roi):
    """Detect green, yellow, and red/orange health bars (including bars washed by golden Heal glow)."""
    left, top, right, bottom = roi
    if right <= left + 8 or bottom <= top + 8:
        return []
    region = image[top:bottom, left:right]
    pre_reg = pre_bgr[top:bottom, left:right]
    hue, sat, val = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HSV))
    dark_strict = val < 92
    dark_soft = val < 116
    diff_pre = np.abs(region.astype(np.int16) - pre_reg.astype(np.int16)).mean(axis=2)

    # (color_name, fill_mask, dark_mask, min_outline, min_diff_pre, max_w, max_h)
    specs = (
        ("green", (hue >= 44) & (hue <= 70) & (sat >= 140) & (val >= 130), dark_strict, 0.22, 16.0, 45, 5),
        # Golden Heal spell shifts green bars slightly yellower (H 37..44), lowers S, and brightens the dark border.
        ("green", (hue >= 37) & (hue <= 70) & (sat >= 105) & (val >= 175), dark_soft, 0.28, 22.0, 45, 6),
        ("yellow", (hue >= 19) & (hue <= 36) & (sat >= 150) & (val >= 165), dark_strict, 0.34, 16.0, 28, 4),
        ("red", ((hue <= 18) | (hue >= 170)) & (sat >= 155) & (val >= 135), dark_strict, 0.36, 16.0, 26, 4),
    )
    bars = []
    for color_name, fill, dark_mask, min_outline, min_diff, max_w, max_h in specs:
        _, _, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8))
        for x, y, w, h, area in stats[1:]:
            if not (5 <= w <= max_w and 1 <= h <= max_h and w >= 2.0 * h and area >= 0.52 * w * h):
                continue
            if y < 2 or y + h + 3 > region.shape[0]:
                continue
            outline = max(dark_mask[y - 2:y, x:x + w].mean(), dark_mask[y + h:y + h + 3, x:x + w].mean())
            if outline < min_outline:
                continue
            if diff_pre[y:y + h, x:x + w].mean() < min_diff:
                continue
            bx, by = int(left + x), int(top + y)
            # Deduplicate overlapping bars between strict and Heal-washed green passes
            if any(abs(bx - ex) <= 6 and abs(by - ey) <= 4 for ex, ey, _, _, _ in bars):
                continue
            bars.append((bx, by, int(w), int(h), color_name))
    return bars


class _Kalman2D:
    """Damped constant-velocity 2D Kalman filter."""

    def __init__(self, pos, vel=(0.0, 0.0), pos_std=18.0, vel_std=45.0, damping=0.8):
        self.state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=float)
        self.cov = np.diag([pos_std ** 2, pos_std ** 2, vel_std ** 2, vel_std ** 2])
        self.damping = damping
        self.last_t = None

    def predict(self, t, q_pos=12.0, q_vel=45.0):
        if self.last_t is None:
            self.last_t = t
            return self.state[:2].copy()
        dt = max(0.0, t - self.last_t)
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        self.state = F @ self.state
        self.state[2:] *= math.exp(-self.damping * dt)
        Q = np.diag([q_pos ** 2, q_pos ** 2, q_vel ** 2, q_vel ** 2]) * dt
        self.cov = F @ self.cov @ F.T + Q
        self.last_t = t
        return self.state[:2].copy()

    def update(self, meas, sigma=12.0, max_speed=85.0):
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        z = np.asarray(meas, dtype=float)
        y = z - H @ self.state
        R = np.eye(2) * (sigma ** 2)
        S = H @ self.cov @ H.T + R
        K = np.linalg.solve(S, H @ self.cov).T
        self.state = self.state + K @ y
        speed = np.linalg.norm(self.state[2:])
        if speed > max_speed:
            self.state[2:] *= max_speed / speed
        I_KH = np.eye(4) - K @ H
        self.cov = I_KH @ self.cov @ I_KH.T + K @ R @ K.T


class HealingWaveTracker:
    """Dual-layer main-army tracker fusing Grand Warden aura, full-spectrum troop bars, and local motion."""

    def __init__(self, deployed_at, commands, layout, pre_bgr=None):
        seed, speed = opening_prior(commands, layout, *VIEWPORT)
        self.deployed_at = deployed_at
        self.pre_bgr = pre_bgr
        if pre_bgr is not None:
            if pre_bgr.shape[:2] != VIEWPORT[::-1]:
                self.pre_bgr = cv2.resize(pre_bgr, VIEWPORT, interpolation=cv2.INTER_AREA)
            self.pre_small = cv2.resize(self.pre_bgr, (600, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
            gw_pts = next((c["points"] for c in commands if c["unit"] == "grand_warden" and c.get("points")), None)
            warden_seed = (np.median(gw_pts, axis=0) * [VIEWPORT[0] / 1800, VIEWPORT[1] / 1440]
                           if gw_pts else seed.copy())
            self.warden_seed = warden_seed.copy()
            self.army_seed = seed.copy()
            self.warden_kf = _Kalman2D(warden_seed, pos_std=20.0, vel_std=35.0, damping=1.0)
            self.warden_seen = None
            self.warden_psr = 0.0
            self.warden_empty_since = None
            self.army_kf = _Kalman2D(seed, speed, pos_std=22.0, vel_std=55.0, damping=0.8)
            self.army_bar_seen = None
            self.army_motion_seen = None
            self.army_radius = 45.0
            self.offset = seed - warden_seed
            self.prev_small = None
            self.motion_ema = None
        else:
            # Legacy fallback when no pre-battle reference image is provided.
            self.tracks = DeployedWaveTracks(deployed_at, seed, speed, observation_gap=.62,
                                             max_initial_reacquire_distance=MAX_INITIAL_WAVE_REACQUIRE_DISTANCE)
            self.center = seed.copy()
            self.previous_frames = deque()
            self.recent_bars = deque(maxlen=2)
            self.bar_tracks = HealthBarTracks(retain_seconds=.35, min_match_margin=4)
            self.bar_anchors = {}
            self.last_observed = None
            self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    def _detect_warden(self, fr_small, pre_small, now):
        pred = self.warden_kf.predict(now, q_pos=10.0, q_vel=30.0)
        diff = fr_small - pre_small
        # Grand Warden's aura rim is a BRIGHT glowing white-lime ring (all 3 channels brighten AND absolute G >= 150).
        # Requiring absolute brightness (fr_g >= 150) + min(db, dg, dr) cleanly rejects:
        #   - destroyed dark walls turning into dull brown/gray rubble (fr_g < 145)
        #   - destroyed buildings revealing green grass (db ~ 0)
        #   - golden yellow Heal spell (db <= 0)
        #   - cyan/blue Freeze & Ice Golem frost rings (db >> dg and dr <= 0)
        #   - magenta Rage rings (dg <= 0)
        db, dg, dr = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
        bright_aura = (fr_small[:, :, 1] >= 150.0) & (fr_small[:, :, 1] >= fr_small[:, :, 0] - 10.0)
        pos_map = np.clip(
            np.minimum(np.minimum(db, dg), dr) - np.maximum(0.0, db - dg) * 1.6,
            0.0,
            45.0,
        ) / 45.0
        pos_map *= bright_aura.astype(np.float32)
        pos_map[:35, :] = 0
        pos_map[410:, :] = 0
        # Mask out the bottom-right 1x speed button & destruction HUD so they never create false ring peaks
        pos_map[310:, 520:] = 0
        res = cv2.matchTemplate(pos_map.astype(np.float32), WARDEN_RING_TEMPLATE, cv2.TM_CCORR)
        th, tw = WARDEN_RING_TEMPLATE.shape
        # Restrict candidate Warden centers strictly inside the playable battlefield diamond (no forest!)
        ry_g, rx_g = np.mgrid[:res.shape[0], :res.shape[1]]
        cx_full = (rx_g + tw // 2) * 2.0
        cy_full = (ry_g + th // 2) * 2.0
        in_diamond = (np.abs(cx_full - 600.0) / 520.0 + np.abs(cy_full - 445.0) / 365.0) <= 1.05
        # Physics reachability bound from Warden's spawn point during opening 12s (Warden walks ~35px/s):
        # guarantees a distant funnel/Freeze/rubble ring in the base center can NEVER steal initial Warden lock!
        elapsed_dep = max(0.0, now - self.deployed_at)
        if elapsed_dep < 12.0:
            max_from_spawn = 95.0 + 48.0 * elapsed_dep
            dist_spawn = np.hypot(cx_full - self.warden_seed[0], cy_full - self.warden_seed[1])
            in_diamond = in_diamond & (dist_spawn <= max_from_spawn)
        valid_vals = res[in_diamond]
        mean_v = float(valid_vals.mean()) if valid_vals.size else float(res.mean())
        std_v = (float(valid_vals.std()) if valid_vals.size else float(res.std())) + 1e-6
        res_masked = np.where(in_diamond, res, -1e9).astype(np.float32)

        gap = (now - self.warden_seen) if self.warden_seen is not None else max(0.0, now - self.deployed_at)
        rad = min(180, 48 + 55 * max(0.1, gap))
        px = int(round(pred[0] / 2)) - tw // 2
        py = int(round(pred[1] / 2)) - th // 2
        r = int(round(rad / 2))
        x1, x2 = max(0, px - r), min(res.shape[1], px + r + 1)
        y1, y2 = max(0, py - r), min(res.shape[0], py + r + 1)
        det, psr = None, 0.0
        if x2 > x1 and y2 > y1:
            sub = res_masked[y1:y2, x1:x2]
            _, maxv, _, loc = cv2.minMaxLoc(sub)
            if maxv > -1e8:
                mx, my = x1 + loc[0], y1 + loc[1]
                psr = (maxv - mean_v) / std_v
                det = np.array([(mx + tw // 2) * 2.0, (my + th // 2) * 2.0])

        # Always evaluate global peak inside the valid diamond (costs ~0.05ms since matchTemplate already ran):
        # 1) Global reacquire if local peak is weak (psr < 5.0), OR
        # 2) Domination override: if the real glowing Warden elsewhere has dramatically higher PSR
        #    (gpsr >= 8.5 and gpsr >= psr + 2.8), snap immediately so a static rubble ring (psr ~ 5-6)
        #    can NEVER trap Warden while the real PSR 12-18 aura walks away!
        _, gmax, _, (gx, gy) = cv2.minMaxLoc(res_masked)
        snapped_global = False
        if gmax > -1e8:
            gpsr = (gmax - mean_v) / std_v
            gdet = np.array([(gx + tw // 2) * 2.0, (gy + th // 2) * 2.0])
            if gpsr >= 8.5 and gpsr >= psr + 2.8:
                det, psr = gdet, gpsr
                snapped_global = True
            elif psr < 5.0 and (self.warden_seen is None or gap >= 0.6):
                max_jump = 125.0 if self.warden_seen is None else min(170.0, 55.0 + 55.0 * gap)
                if gpsr >= 6.2 and np.linalg.norm(gdet - pred) <= max_jump:
                    det, psr = gdet, gpsr

        self.warden_psr = psr
        if det is not None and psr >= 4.5:
            if snapped_global and np.linalg.norm(det - pred) > 90.0:
                # Re-initialize KF cleanly at the dominating global Warden peak (no huge fake velocity spike!)
                self.warden_kf = _Kalman2D(det, pos_std=18.0, vel_std=35.0, damping=1.0)
                self.warden_kf.last_t = now
                self.warden_seen = now
                return self.warden_kf.state[:2].copy(), True
            if self.warden_seen is None or np.linalg.norm(det - pred) <= max(45.0, 58.0 * max(0.1, gap)):
                self.warden_kf.update(det, sigma=10.0 if psr >= 6.5 else 16.0, max_speed=65.0)
                self.warden_seen = now
                return self.warden_kf.state[:2].copy(), True
        return self.warden_kf.state[:2].copy(), (self.warden_seen is not None and (now - self.warden_seen) <= 2.2)

    @staticmethod
    def _cluster_bars(points, max_dist=78.0):
        n = len(points)
        if n == 0:
            return []
        pts = np.asarray(points, dtype=float)
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(n):
                    if not visited[v] and np.linalg.norm(pts[u] - pts[v]) <= max_dist:
                        visited[v] = True
                        stack.append(v)
            clusters.append(comp)
        return clusters

    def _motion_centroid(self, fr_small, pre_small, warden_pos, warden_active, army_pred):
        if self.prev_small is None:
            return None
        frame_diff = np.abs(fr_small - self.prev_small).mean(axis=2)
        pre_diff = np.abs(fr_small - pre_small).mean(axis=2)
        # Softly saturate motion per pixel in [0, 1] so bright explosions/flashes don't outweigh moving troops!
        active = np.clip((frame_diff - 12.0) / 18.0, 0.0, 1.0) * (pre_diff > 16.0)
        # Mask out bright white/silver UI popups (1-star / 2-star "爽!" banner!) & blue Freeze / magenta Rage flashes
        hsv_s = cv2.cvtColor(np.clip(fr_small, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        b_ch, g_ch, r_ch = fr_small[:, :, 0], fr_small[:, :, 1], fr_small[:, :, 2]
        popup_or_flash = (
            ((hsv_s[:, :, 2] > 228) & (hsv_s[:, :, 1] < 48))  # silver star banner & white flash cores
            | ((b_ch - g_ch) > 28.0)                           # cyan/blue Freeze & Ice explosions
            | (((r_ch - g_ch) > 32.0) & ((b_ch - g_ch) > 20.0))  # magenta Rage flashes
        )
        active[popup_or_flash] = 0.0
        active[:35, :] = 0.0
        active[410:, :] = 0.0

        yy, xx = np.ogrid[:480, :600]
        if warden_active:
            wx, wy = warden_pos[0] / 2.0, warden_pos[1] / 2.0
            # Mask out Warden's feet and upper body/cape (which extends ~25px above aura center).
            active[((xx - wx) / 14.0) ** 2 + ((yy - (wy - 12.0)) / 22.0) ** 2 < 1.0] = 0.0
            disk = (xx - wx) ** 2 + (yy - wy) ** 2 <= 80 ** 2
        else:
            ax, ay = army_pred[0] / 2.0, army_pred[1] / 2.0
            disk = (xx - ax) ** 2 + (yy - ay) ** 2 <= 65 ** 2
        active[~disk] = 0.0

        # Reject sudden frame-wide explosion/star-banner bursts (>35% of search disk lighting up at once)
        disk_area = max(1, int(np.count_nonzero(disk)))
        if float(np.count_nonzero(active > 0.25)) / disk_area > 0.35:
            return None

        blurred = cv2.GaussianBlur(active.astype(np.float32), (0, 0), 13.0)
        # Temporal EMA suppresses 1-frame projectiles/explosions while preserving steady troop movement
        if self.motion_ema is None or self.motion_ema.shape != blurred.shape:
            self.motion_ema = blurred.copy()
        else:
            self.motion_ema = 0.55 * self.motion_ema + 0.45 * blurred

        ax, ay = army_pred[0] / 2.0, army_pred[1] / 2.0
        prior = np.exp(-0.5 * ((xx - ax) ** 2 + (yy - ay) ** 2) / (45.0 ** 2))
        score_map = self.motion_ema * (0.25 + 0.75 * prior)
        _, maxv, _, (mx, my) = cv2.minMaxLoc(score_map)
        if maxv < 0.045:
            return None
        return np.array([mx * 2.0, my * 2.0])

    def observe(self, frame, now, transform=None, warden_alive=True):
        if self.pre_bgr is None:
            return self._observe_legacy(frame, now)
        if now < self.deployed_at:
            return []

        if transform is not None and not np.allclose(transform, [[1, 0, 0], [0, 1, 0]], atol=0.5):
            pre_aligned = cv2.warpAffine(self.pre_bgr, transform, VIEWPORT)
            pre_small = cv2.resize(pre_aligned, (600, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
        else:
            pre_aligned = self.pre_bgr
            pre_small = self.pre_small

        fr_small = cv2.resize(frame, (600, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
        if warden_alive:
            warden_pos, warden_active = self._detect_warden(fr_small, pre_small, now)
        else:
            warden_pos, warden_active = None, False

        army_pred = self.army_kf.predict(now, q_pos=14.0, q_vel=45.0)
        elapsed_dep = max(0.0, now - self.deployed_at)
        bar_gap = (now - self.army_bar_seen) if self.army_bar_seen is not None else elapsed_dep
        if warden_active and self.army_bar_seen is not None and bar_gap > 0.35:
            army_pred = 0.65 * army_pred + 0.35 * (warden_pos + self.offset)

        # Only expand to full-battlefield bar scan AFTER the main army has actually been locked at least once
        # and lost bars for >=2.0s (NEVER during opening 0..6s while funnel troops across the map are dying!).
        if self.army_bar_seen is not None and elapsed_dep >= 6.0 and bar_gap >= 2.0:
            roi = (60, 75, VIEWPORT[0] - 60, 820)
        else:
            anchors = [army_pred]
            if warden_active:
                anchors.append(warden_pos)
            # Keep opening ROI tight around deployment line so distant funnel troops (Queen/Siege) cannot hijack!
            rad_roi = (195 if elapsed_dep < 6.0
                       else (225 if warden_active else min(270, 175 + 35 * max(0.0, bar_gap))))
            lo = np.rint(np.min(anchors, axis=0) - rad_roi).astype(int)
            hi = np.rint(np.max(anchors, axis=0) + rad_roi).astype(int)
            roi = (max(0, lo[0]), max(75, lo[1]), min(VIEWPORT[0], hi[0]), min(820, hi[1]))

        raw_bars = detect_troop_health_bars(frame, pre_aligned, roi)
        heroes = [b for b in raw_bars if has_level_badge(frame, b[:4])]
        all_troops = []
        for x, y, w, h, cname in raw_bars:
            if any(abs(x - hb[0]) <= 28 and abs(y - hb[1]) <= 11 for hb in heroes):
                continue
            pt = np.array([x + w / 2.0, y + h + 12.0])
            all_troops.append((pt, cname))

        # Track whether Warden's detected aura actually has any friendly bars nearby (detects hijacked Warden!)
        if warden_active:
            has_bar_near_warden = any(
                np.linalg.norm(p - warden_pos) <= 185.0 for p, _ in all_troops
            ) or any(
                np.linalg.norm(np.array([hb[0] + hb[2] / 2.0, hb[1] + hb[3] + 12.0]) - warden_pos) <= 185.0
                for hb in heroes
            )
            if has_bar_near_warden:
                self.warden_empty_since = None
            elif self.warden_empty_since is None:
                self.warden_empty_since = now
        else:
            self.warden_empty_since = None
        warden_starved = (
            warden_active
            and self.warden_empty_since is not None
            and (now - self.warden_empty_since) >= 1.5
        )

        source = None
        selected_bars = []
        if all_troops:
            pts = [p for p, _ in all_troops]
            clusters = self._cluster_bars(pts, max_dist=84.0)
            best_score, best_comp = -1e9, None
            for comp in clusters:
                c_pts = np.asarray([pts[i] for i in comp])
                c_center = c_pts.mean(axis=0)
                d_army = float(np.linalg.norm(c_center - army_pred))
                d_warden = float(np.linalg.norm(c_center - warden_pos)) if warden_active else 0.0
                # Physics reachability gate during opening 8s: never jump onto funnel troops across the map!
                if elapsed_dep < 8.0 and np.linalg.norm(c_center - self.army_seed) > (130.0 + 55.0 * elapsed_dep):
                    continue
                # Even during full-battlefield reacquire, cap teleport jump so distant cleanup/funnel clusters can't hijack
                if d_army > min(280.0, 145.0 + 45.0 * bar_gap):
                    continue
                # Filter isolated 1-2 bar flank noise (e.g. Siege Wizards) when Warden is healthy & non-starved,
                # but NEVER throw away a real main-army cluster of >=3 troop bars!
                if len(comp) == 1 and (d_army > 110 or (warden_active and not warden_starved and d_warden > 155)):
                    continue
                if len(comp) == 2 and warden_active and not warden_starved and d_warden > 175 and d_army > 140:
                    continue
                w_pen = 0.0 if warden_starved else (0.40 * d_warden)
                a_pen = 0.20 * d_army if (bar_gap >= 1.5 and len(comp) >= 3) else 0.50 * d_army
                score = 30.0 * min(len(comp), 6) - a_pen - w_pen
                if score > best_score:
                    best_score, best_comp = score, comp
            if best_comp is not None:
                c_pts = np.asarray([pts[i] for i in best_comp])
                c_center = c_pts.mean(axis=0)
                spread = float(np.linalg.norm(c_pts - c_center, axis=1).max()) if len(best_comp) > 1 else 25.0
                sigma = 10.0 if len(best_comp) >= 3 else (15.0 if len(best_comp) == 2 else 22.0)
                self.army_kf.update(c_center, sigma=sigma, max_speed=95.0)
                self.army_bar_seen = now
                self.army_radius = 0.7 * self.army_radius + 0.3 * float(np.clip(spread + 20.0, 28.0, 75.0))
                selected_bars = [all_troops[i] for i in best_comp]
                source = "health_bars" if len(best_comp) >= 2 else "single_bar"
                # Self-heal hijacked Warden: only after opening 6s, if >=3 troop bars appear far from a starved Warden
                if elapsed_dep >= 6.0 and warden_starved and len(best_comp) >= 3 and np.linalg.norm(c_center - warden_pos) > 185.0:
                    self.warden_kf = _Kalman2D(c_center, pos_std=35.0, vel_std=35.0, damping=1.0)
                    self.warden_seen = None
                    self.warden_empty_since = None
                    warden_pos, warden_active = None, False
                elif warden_active:
                    new_off = self.army_kf.state[:2] - warden_pos
                    nrm = np.linalg.norm(new_off)
                    if nrm > 155.0:
                        new_off *= 155.0 / nrm
                    self.offset = 0.72 * self.offset + 0.28 * new_off

        if source is None:
            # Prefer reliable Warden aura anchor over noisy raw motion; tighten motion step (<=60px) and damping
            # so explosions/star-banners/flank Giants can NEVER cause runaway drift!
            m_pos = self._motion_centroid(fr_small, pre_small, warden_pos, warden_active and not warden_starved, army_pred)
            motion_ok = (
                m_pos is not None
                and np.linalg.norm(m_pos - army_pred) <= 60.0
                and (not warden_active or warden_starved or np.linalg.norm(m_pos - warden_pos) <= 145.0)
            )
            if warden_active and not warden_starved and self.warden_psr >= 5.8:
                if motion_ok:
                    fused = 0.55 * (warden_pos + self.offset) + 0.45 * m_pos
                    self.army_kf.update(fused, sigma=34.0, max_speed=55.0)
                    self.army_motion_seen = now
                    source = "warden_anchor"
                else:
                    self.army_kf.update(warden_pos + self.offset, sigma=30.0, max_speed=55.0)
                    self.offset *= 0.96
                    source = "warden_anchor"
            elif motion_ok:
                self.army_kf.update(m_pos, sigma=44.0, max_speed=45.0)
                self.army_motion_seen = now
                source = "motion_only"
            elif warden_active and not warden_starved:
                self.army_kf.update(warden_pos + self.offset, sigma=32.0, max_speed=55.0)
                self.offset *= 0.96
                source = "warden_anchor"
            elif self.army_bar_seen is not None and bar_gap <= 2.5:
                source = "predicted"

        self.prev_small = fr_small
        if source is None:
            return []

        age = (now - self.army_bar_seen) if self.army_bar_seen is not None else (now - self.deployed_at)
        uncertainty = float(2 * np.sqrt(np.linalg.eigvalsh(self.army_kf.cov[:2, :2])[-1]))
        group = {
            "id": 1,
            "center": self.army_kf.state[:2].tolist(),
            "velocity": self.army_kf.state[2:].tolist(),
            "radius": float(self.army_radius),
            "source": source,
            "uncertainty_px": round(uncertainty, 2),
            "age_seconds": round(age, 3),
            "indicators": len(selected_bars),
            "bars": [
                {"point": p.tolist(), "yellow": cname in ("yellow", "red"), "color": cname}
                for p, cname in selected_bars
            ],
            "warden": warden_pos.tolist() if warden_active else None,
            "warden_psr": round(self.warden_psr, 2),
        }
        return [group]

    def _observe_legacy(self, frame, now):
        self.previous_frames.append((now, frame))
        while len(self.previous_frames) > 1 and now-self.previous_frames[1][0] >= .30:
            self.previous_frames.popleft()
        previous = self.previous_frames[0]
        active = self.tracks.search_centers(now)
        anchors = [self.center, *active]
        lo = np.rint(np.min(anchors, axis=0)).astype(int)
        hi = np.rint(np.max(anchors, axis=0)).astype(int)
        radius = 130 if active else 182
        roi = (max(0, lo[0]-radius), max(75, lo[1]-radius),
               min(VIEWPORT[0], hi[0]+radius), min(820, hi[1]+radius))
        gap = now-previous[0]
        motion = motion_evidence(previous[1], frame, roi, self.flow) if .19 <= gap <= .8 else None
        observations, visible, yellow = [], [], set()
        for low in (False, True):
            for x, y, w, h in health_bars(frame, roi, yellow=low):
                point = np.array((x+w/2, y+h+12))
                if all(np.linalg.norm(point-anchor) > radius for anchor in anchors):
                    continue
                if any(np.linalg.norm(point-seen) < 4 for seen in visible):
                    continue
                visible.append(point)
                moving = motion is not None and nearby_motion(motion[0], point, roi)
                persistent = bool(self.recent_bars) and all(
                    any(np.linalg.norm(point-old) < 14 for old in older) for older in self.recent_bars
                )
                # A newly seen yellow bar must survive the temporal check before it can trigger healing.
                if not persistent and (low or not moving):
                    continue
                observations.append((point, (x, y, w, h), moving))
                if low:
                    yellow.add(tuple(point))
        self.recent_bars.append(visible)
        heroes, troops, primary = deployment_group(frame, observations)
        if primary:
            self.center = .65*self.center + .35*np.asarray(primary["center"])

        def local_motion(track, age):
            area = {"center": track.state[:2].tolist(), "radius": track.radius}
            # The Kalman flow correction expects a displacement over 0.35s.
            return follow_hidden_wave(area, age, .35*.35/gap, motion, roi, heroes) if motion else None

        measurements = split_group_measurements(troops)
        troop_points = {tuple(point) for point in troops}
        # These bars already passed the motion or persistence check above.
        bar_observations = [
            (point, box[2], True) for point, box, _ in observations if tuple(point) in troop_points
        ]
        period = now-self.last_observed if self.last_observed is not None else .1
        self.last_observed = now
        visible_tracks, _ = self.bar_tracks.update(now, bar_observations, period)
        track_ids = {tuple(bar.point): bar.id for bar in visible_tracks}
        single_bar_positions = self.single_bar_positions(now, visible_tracks)
        groups = self.tracks.update(now, measurements, local_motion, primary,
                                    single_bar_positions=single_bar_positions)
        observed = [group for group in groups if group["source"] == "health_bars"]
        for group in observed:
            group["bars"] = []
        # Only the nearest observed group can claim a visible bar.
        for point in troops:
            if observed:
                group = min(observed, key=lambda row: math.dist(point, row["center"]))
                if math.dist(point, group["center"]) <= max(25, group["radius"]):
                    group["bars"].append({"point": point.tolist(), "yellow": tuple(point) in yellow})
        for group in observed:
            offsets = {
                track_ids[tuple(bar["point"])]: np.subtract(bar["point"], group["center"])
                for bar in group["bars"] if tuple(bar["point"]) in track_ids
            }
            if len(offsets) >= 2:
                self.bar_anchors[group["id"]] = (now, offsets)
            else:
                self.bar_anchors.pop(group["id"], None)
        return groups

    def single_bar_positions(self, now, visible_tracks):
        self.bar_anchors = {group_id: entry for group_id, entry in self.bar_anchors.items()
                            if now-entry[0] <= .8}
        owners = Counter(bar_id for _, offsets in self.bar_anchors.values() for bar_id in offsets)
        continuations = {}
        for group_id, (_, offsets) in self.bar_anchors.items():
            matches = [bar for bar in visible_tracks if bar.id in offsets]
            if len(matches) == 1 and owners[matches[0].id] == 1:
                bar = matches[0]
                continuations[group_id] = (bar.id, bar.point-offsets[bar.id])
        return continuations


@dataclass
class Building:
    id: int
    name: str
    center: np.ndarray
    patches: list
    outer: float
    inner: float


class HealingMap:
    """Map the opening YOLO detections to tile coordinates and check which targets remain."""

    def __init__(self, layout, profiles, grid, reference):
        self.grid = grid
        self.buildings = []
        self.visible_at = {}
        width, height = (layout["viewport"][key] for key in ("width", "height"))
        detections = sorted(layout["detections"], key=lambda d: d["confidence"], reverse=True)
        seen = []
        for index, d in enumerate(detections):
            name, box = d["class"], d["bbox_xyxy"]
            if name not in SPLASH_DEFENSES or name not in profiles:
                continue
            if any(name == previous["class"] and intersection_over_union(box, previous["bbox_xyxy"]) > .28
                   for previous in seen):
                continue
            seen.append(d)
            x1, y1, x2, y2 = box
            x = (x1+x2)*.5*VIEWPORT[0]/width
            y = y2*VIEWPORT[1]/height
            patches = []
            u = round(x/2)
            for height_fraction in (.48, .80):
                v = round((y1+height_fraction*(y2-y1))*VIEWPORT[1]/height/2)
                patch = reference[v-6:v+7, u-6:u+7].copy()
                if not (8 <= u < reference.shape[1]-8 and 8 <= v < reference.shape[0]-8):
                    continue
                if patch.std() >= 10:
                    patches.append(((u, v), patch))
            if not patches:
                continue
            profile = profiles[name]
            ground = (profile.get("kind") == "direct" and "ground" in profile.get("targets", []))
            self.buildings.append(Building(index, name, self.tile((x, y-8)), patches,
                                           profile.get("range", 0) if ground else 0,
                                           profile.get("min_range", 0)))

    def tile(self, point):
        return np.asarray(self.grid.pixel_to_grid(point[0]*1000/VIEWPORT[0], point[1]*800/VIEWPORT[1]))

    def pixel(self, tile, transform):
        x, y = self.grid.grid_to_pixel(*tile)
        return transform @ np.array([x*VIEWPORT[0]/1000, y*VIEWPORT[1]/800, 1.])

    def alive_near(self, positions, transform, gray, now):
        live = []
        for building in self.buildings:
            distance = min(math.dist(building.center, point) for point in positions)
            if distance > building.outer+1:
                continue
            for patch_center, patch in building.patches:
                x, y = np.rint((transform @ np.r_[np.asarray(patch_center)*2, 1])/2).astype(int)
                if not (12 <= x < gray.shape[1]-12 and 12 <= y < gray.shape[0]-12):
                    continue
                similarity = cv2.matchTemplate(gray[y-11:y+12, x-11:x+12], patch,
                                               cv2.TM_CCOEFF_NORMED).max()
                if similarity >= .76:
                    self.visible_at[building.id] = now
                    break
            if now-self.visible_at.get(building.id, -math.inf) <= .9:
                live.append(building)
        return live


def splash_cover(position, buildings):
    return {b.id: b.name for b in buildings if b.inner+.30 <= math.dist(position, b.center) <= b.outer-.30}


class HealingPolicy:
    """Heal a confirmed troop group under splash exposure or with low health."""

    def __init__(self, layout):
        self.layout = layout
        self.yellow_seen = {}
        self.active = []

    def decide(self, now, groups, transform, gray):
        if transform is None:
            self.yellow_seen.clear()
            return None, []
        if not groups:
            return None, []
        inverse = cv2.invertAffineTransform(transform)
        centers = {g["id"]: self.layout.tile(inverse @ np.r_[g["center"], 1]) for g in groups}
        buildings = self.layout.alive_near(centers.values(), transform, gray, now)
        observed = [g for g in groups if g["source"] == "health_bars" and len(g["bars"]) >= 2]
        self.active = [(end, at) for end, at in self.active if now < end]
        self.yellow_seen = {key: at for key, at in self.yellow_seen.items() if now-at <= .8}
        decisions = []
        assessments = []
        for group in observed:
            group_id = group["id"]
            center = centers[group_id]
            health = group["bars"]
            yellow = sum(bar["yellow"] for bar in health)
            low = yellow >= 2 and yellow/len(health) >= .35
            health_trigger = low and group_id in self.yellow_seen
            if low:
                self.yellow_seen[group_id] = now
            else:
                self.yellow_seen.pop(group_id, None)
            members = [self.layout.tile(inverse @ np.r_[bar["point"], 1]) for bar in health]
            covered = sum(math.dist(point, center) <= HEAL_RADIUS-.2 for point in members)
            splash = splash_cover(center, buildings)
            reasons = [name for name, yes in (("yellow_health", health_trigger),
                                               ("in_splash_range", bool(splash))) if yes]
            row = {"group": group_id, "yellow": yellow, "bars": len(health),
                   "tile": center.tolist(), "pixel": self.layout.pixel(center, transform).tolist(),
                   "covered": covered, "splash": list(splash.values()), "reasons": reasons}
            assessments.append(row)
            if any(math.dist(center, at) < HEAL_RADIUS for _, at in self.active):
                row["state"] = "already_healing"
            elif not reasons:
                row["state"] = "waiting"
            else:
                row["state"] = "candidate"
                decisions.append(row)
        return max(decisions, key=lambda row: (row["covered"], len(row["reasons"]), row["yellow"]),
                   default=None), assessments

    def record_cast(self, now, decision):
        self.active.append((now+HEAL_DURATION, decision["tile"]))
