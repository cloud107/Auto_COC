"""Stage 1 Potential-Field FSM outputting factorized (a_slot, a_grid) actions with legal masking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from battle_grid_state import GRID_SIZE, NUM_SLOTS
from healing_spell import HEAL_DURATION, HEAL_RADIUS

SLOTS = {
    "miner_squad": 1,
    "hog_squad": 2,
    "ice_golem": 3,
    "archer": 4,
    "druid": 5,
    "siege_barracks": 6,
    "dragon_duke": 7,
    "archer_queen": 8,
    "grand_warden": 9,
    "royal_champion": 10,
    "healing_spell": 11,
    "freeze_spell": 12,
    "totem_spell": 13,
    "ice_block": 14,
    "poison_spell": 15,
    "revive_spell": 16,
}

SPELL_SLOTS = {11, 12, 13, 14, 15, 16}
HERO_SLOTS = {7, 8, 9, 10}


@dataclass
class FactorizedAction:
    """Factorized action contract: slot in 0..16 (0=No-Op), grid in 44x44."""

    slot: int
    grid: tuple[int, int] | None  # (col, row) in 0..43
    kind: str  # "noop" | "deploy" | "spell" | "hero_ability"
    reason: str
    phase: str
    slot_mask: np.ndarray | None = None   # bool[17]
    grid_mask: np.ndarray | None = None   # bool[44, 44]

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "grid": list(self.grid) if self.grid is not None else None,
            "kind": self.kind,
            "reason": self.reason,
            "phase": self.phase,
        }


def compute_action_masks(
    card_counts: dict[int, int],
    hero_states: list[int],
    deployable_mask: np.ndarray,
    selected_slot: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute legal slot mask (17,) and spatial grid mask (44, 44) for BC/RL compatibility."""
    slot_mask = np.zeros(NUM_SLOTS + 1, dtype=bool)
    slot_mask[0] = True  # No-Op is always legal
    hero_slot_order = (7, 8, 9, 10)
    for slot in range(1, NUM_SLOTS + 1):
        if slot in hero_slot_order:
            h_idx = hero_slot_order.index(slot)
            # Legal to deploy (state==0) or trigger ability (state==1)
            slot_mask[slot] = hero_states[h_idx] in (0, 1)
        else:
            slot_mask[slot] = card_counts.get(slot, 0) > 0

    if selected_slot is None or selected_slot == 0 or selected_slot in SPELL_SLOTS:
        grid_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    else:
        grid_mask = deployable_mask >= 0.5
    return slot_mask, grid_mask


def score_entry_sides(spatial: np.ndarray, buildings: list | None = None) -> dict:
    """Evaluate all 4 outer edges using potential-field cost (TH proximity, Firespitter cones, Monolith safety)."""
    th_ch = spatial[0]
    core_ch = spatial[1]
    deploy_ch = spatial[7]
    if th_ch.max() <= 0:
        th_pos = np.array([GRID_SIZE / 2.0, GRID_SIZE / 2.0])
    else:
        r, c = np.unravel_index(int(np.argmax(th_ch)), th_ch.shape)
        th_pos = np.array([float(c) + 0.5, float(r) + 0.5])

    deploy_rows, deploy_cols = np.where(deploy_ch >= 0.5)
    if len(deploy_rows) == 0:
        raise ValueError("No legal deployable cells in Ch 7")

    min_c, max_c = int(deploy_cols.min()), int(deploy_cols.max())
    min_r, max_r = int(deploy_rows.min()), int(deploy_rows.max())
    candidates = []
    # Define 4 sides by proximity to outer deployable bounds
    side_specs = (
        ("north", [(int(c), int(r)) for r, c in zip(deploy_rows, deploy_cols) if r <= min_r + 2], 0),
        ("south", [(int(c), int(r)) for r, c in zip(deploy_rows, deploy_cols) if r >= max_r - 2], 0),
        ("west",  [(int(c), int(r)) for r, c in zip(deploy_rows, deploy_cols) if c <= min_c + 2], 1),
        ("east",  [(int(c), int(r)) for r, c in zip(deploy_rows, deploy_cols) if c >= max_c - 2], 1),
    )

    side_map = {name: cells for name, cells, _ in side_specs}
    # Adjacent perpendicular edge on the OPPOSITE side from queen_flank (flanks[1] is +axis, flanks[0] is -axis),
    # placed ~11 tiles down the adjacent edge (e.g. 11 o'clock for north entry) so Dragon Duke clears
    # along the flank AWAY from the main push instead of merging into the main army.
    wrap_specs = {
        "east":  ("north", 0, max_c - 10),  # flanks[0] is north (-row), flanks[1] is south (+row)
        "west":  ("north", 0, min_c + 3),   # 12 o'clock corner so Dragon clears north side cleanly before west push!
        "north": ("west",  1, min_r + 12),  # 11 o'clock on west edge -> clears downward away from north main wave!
        "south": ("west",  1, max_r - 12),  # flanks[0] is west  (-col), flanks[1] is east  (+col)
    }

    monoliths = [b for b in (buildings or []) if b.alive and b.name == "monolith"]
    guardians = [b for b in (buildings or []) if b.alive and b.name.startswith("guardian_")]
    firespitters = [
        b for b in (buildings or [])
        if b.alive and b.name == "firespitter" and b.facing_rad is not None
    ]

    for side_name, cells, axis in side_specs:
        by_axis = sorted(cells, key=lambda cell: cell[axis])
        if len(by_axis) < 12 or by_axis[-1][axis] - by_axis[0][axis] < 15:
            continue
        center_idx = min(
            range(5, len(by_axis) - 5),
            key=lambda i: math.hypot(by_axis[i][0] + 0.5 - th_pos[0], by_axis[i][1] + 0.5 - th_pos[1]),
        )
        center = by_axis[center_idx]
        flanks = [
            min(by_axis, key=lambda cell: abs(cell[axis] - (center[axis] + off)))
            for off in (-7, 17)  # push queen_flank far out to the opposite corner
        ]
        siege_cell = min(by_axis, key=lambda cell: abs(cell[axis] - (center[axis] + 10)))
        if min(abs(cell[axis] - center[axis]) for cell in flanks) < 5:
            continue
        # Wrap Dragon flank past its corner onto the adjacent perpendicular edge when available
        dragon_cell = flanks[0]
        queen_cell = flanks[1]
        adj_name, adj_axis, adj_target = wrap_specs[side_name]
        adj_cells = side_map.get(adj_name, [])
        if adj_cells:
            dragon_cell = min(adj_cells, key=lambda cell: abs(cell[adj_axis] - adj_target))

        # Monolith safety for Dragon Duke: Monolith deals % max-HP single-target damage that melts Dragon Duke.
        # If an exterior/offset Monolith sits near dragon_cell, swap Dragon <-> Queen flanks so 2x Ice Golem + Queen
        # tank the Monolith side and Dragon funnels the Monolith-free corner!
        monolith_penalty = 0.0
        if monoliths:
            d_dragon_mono = min(math.hypot(dragon_cell[0] - m.tile[0], dragon_cell[1] - m.tile[1]) for m in monoliths)
            d_queen_mono = min(math.hypot(queen_cell[0] - m.tile[0], queen_cell[1] - m.tile[1]) for m in monoliths)
            if d_dragon_mono <= 13.5 and d_queen_mono > d_dragon_mono + 2.5:
                dragon_cell, queen_cell = queen_cell, dragon_cell
                d_dragon_mono = d_queen_mono
            if d_dragon_mono <= 12.5:
                monolith_penalty = 8.0 * (1.0 - d_dragon_mono / 12.5) + 3.0

        # Guardian flank safety: if the Guardian is offset/exterior, NEVER send solo Dragon Duke or Queen
        # to funnel into the Guardian's corner (they can't tank the Guardian alone).
        guardian_flank_penalty = 0.0
        if guardians:
            d_dragon_g = min(math.hypot(dragon_cell[0] - g.tile[0], dragon_cell[1] - g.tile[1]) for g in guardians)
            d_queen_g = min(math.hypot(queen_cell[0] - g.tile[0], queen_cell[1] - g.tile[1]) for g in guardians)
            min_flank_g = min(d_dragon_g, d_queen_g)
            if min_flank_g <= 22.0:
                guardian_flank_penalty = 12.0 * (1.0 - min_flank_g / 22.0) + 4.0

        main_line = [
            min(by_axis, key=lambda cell: abs(cell[axis] - (center[axis] + off)))
            for off in (-2, -1, 1, 2)
        ]
        # Sample core defense potential & head-on Firespitter cone exposure along center -> Town Hall
        dist_th = math.hypot(center[0] + 0.5 - th_pos[0], center[1] + 0.5 - th_pos[1])
        corridor_threat = 0.0
        firespitter_headon = 0.0
        for alpha in np.linspace(0.0, 1.0, 12):
            sx = center[0] + alpha * (th_pos[0] - center[0])
            sy = center[1] + alpha * (th_pos[1] - center[1])
            gx = int(np.clip(round(sx), 0, GRID_SIZE - 1))
            gy = int(np.clip(round(sy), 0, GRID_SIZE - 1))
            corridor_threat += float(core_ch[gy, gx])
            for fs in firespitters:
                if fs.covers_tile((sx, sy)):
                    firespitter_headon += 0.70

        # Lower score is better: closer to TH, flanking behind Firespitter cones, and keeping flanks away from Monolith/Guardian
        score = dist_th + 0.35 * corridor_threat + firespitter_headon + monolith_penalty + guardian_flank_penalty
        candidates.append({
            "side": side_name,
            "score": round(score, 2),
            "dist_th": round(dist_th, 2),
            "corridor_threat": round(corridor_threat, 2),
            "firespitter_headon": round(firespitter_headon, 2),
            "monolith_penalty": round(monolith_penalty, 2),
            "guardian_flank_penalty": round(guardian_flank_penalty, 2),
            "dragon_flank": dragon_cell,
            "queen_flank": queen_cell,
            "siege_flank": siege_cell,
            "center": center,
            "main_line": main_line,
        })
    if not candidates:
        raise ValueError("No outer side offers two flanks and a main entry corridor")
    return min(candidates, key=lambda item: item["score"])


FREEZE_PRIORITY_DEFENSES = {
    "monolith": 4.0,
    "scattershot": 3.5,
    "inferno_tower": 3.5,
    "firespitter": 3.0,
    "revenge_tower": 2.8,
    "town_hall": 2.5,
    "firecracker_tower": 2.2,
    "super_wizard_tower": 2.0,
}

# Additive splash-threat weights for Healing Spell placement (summed over alive defenses covering army_tile).
# A single exterior Mortar (0.4) or isolated tower never wastes a Heal, while real splash clusters (>= 3.2) trigger cleanly.
HEAL_SPLASH_WEIGHTS = {
    "firespitter": 2.2,         # uses real 150° directional cone when detected
    "scattershot": 2.0,
    "firecracker_tower": 2.0,
    "super_wizard_tower": 1.8,
    "spiky_tower": 1.8,
    "inferno_tower": 1.5,
    "bomb_tower": 1.2,
    "mortar": 0.4,
}


@dataclass
class PotentialFieldFSM:
    """4-state potential-field FSM emitting factorized (slot, grid) actions at 10Hz."""

    funnel_wait_s: float = 12.0
    warden_min_delay_s: float = 10.0
    warden_min_push_tiles: float = 8.0
    enable_freeze: bool = False
    enable_totem: bool = False
    enable_warden_ability: bool = False
    enable_rc_ability: bool = False
    enable_cleanup: bool = False
    enable_revive: bool = False
    phase: str = field(default="SCOUT_AND_FUNNEL", init=False)
    plan: dict | None = field(default=None, init=False)
    entry_center: tuple[float, float] | None = field(default=None, init=False)
    funnel_started_at: float | None = field(default=None, init=False)
    main_pushed_at: float | None = field(default=None, init=False)
    warden_tome_end: float = field(default=-1.0, init=False)
    rc_ability_end: float = field(default=-1.0, init=False)
    last_heal_at: float = field(default=-99.0, init=False)
    last_endgame_dump_at: float = field(default=-99.0, init=False)
    active_spells: list[dict] = field(default_factory=list, init=False)
    yellow_seen_at: float | None = field(default=None, init=False)
    bar_history: list[tuple[float, int]] = field(default_factory=list, init=False)
    hero_hp_history: dict[int, list[tuple[float, float]]] = field(default_factory=dict, init=False)
    pending_queue: list[tuple[int, tuple[int, int], str]] = field(default_factory=list, init=False)

    def _enqueue_opening(self, spatial: np.ndarray, now: float):
        self.plan = score_entry_sides(spatial)
        self.pending_queue = [
            (SLOTS["dragon_duke"], self.plan["dragon_flank"], "funnel_dragon"),
            (SLOTS["ice_golem"], self.plan["queen_flank"], "funnel_queen_tank"),
            (SLOTS["archer_queen"], self.plan["queen_flank"], "funnel_queen"),
            (SLOTS["siege_barracks"], self.plan["queen_flank"], "funnel_siege"),
        ]
        self.funnel_started_at = now

    def _enqueue_main_push(self, now: float):
        assert self.plan is not None
        line = self.plan["main_line"]
        center = self.plan["center"]
        self.pending_queue = [
            *((SLOTS["miner_squad"], cell, "main_miner") for cell in line),
            (SLOTS["hog_squad"], line[1], "main_hog"),
            (SLOTS["hog_squad"], line[2], "main_hog"),
            (SLOTS["grand_warden"], center, "main_warden"),
            (SLOTS["royal_champion"], center, "main_rc"),
            (SLOTS["druid"], line[1], "main_druid"),
            (SLOTS["druid"], line[2], "main_druid"),
        ]
        self.main_pushed_at = now

    def record_spell(self, now: float, slot: int, tile: tuple[int, int]):
        if slot == SLOTS["healing_spell"]:
            self.active_spells.append({
                "slot": slot, "tile": np.asarray(tile, dtype=float),
                "radius": HEAL_RADIUS, "duration": HEAL_DURATION, "end": now + HEAL_DURATION,
            })
        elif slot == SLOTS["freeze_spell"]:
            self.active_spells.append({
                "slot": slot, "tile": np.asarray(tile, dtype=float),
                "radius": 3.5, "duration": 5.5, "end": now + 5.5,
            })

    def _update_hero_hp_drop(self, now: float, slot: int, hp: float | None) -> tuple[float | None, float]:
        """Track rolling 1.6s HUD HP history for a hero and return (current_hp, recent_drop)."""
        if hp is None or hp <= 0.0:
            return None, 0.0
        hist = self.hero_hp_history.setdefault(slot, [])
        hist.append((now, hp))
        hist[:] = [(ts, val) for ts, val in hist if now - ts <= 1.6]
        peak = max((val for _, val in hist), default=hp)
        return hp, max(0.0, peak - hp)

    def step(
        self,
        state: dict,
        card_counts: dict[int, int],
        hero_states: list[int],
        army_groups: list[dict] | None = None,
        enemy_hero_tiles: list[np.ndarray] | None = None,
        fallen_hero_tile: np.ndarray | None = None,
        live_guardian_tile: np.ndarray | None = None,
        guardian_awake: bool = False,
        hero_hps: dict[int, float] | None = None,
    ) -> FactorizedAction:
        now = state["time"]
        spatial = state["spatial"]
        self.active_spells = [sp for sp in self.active_spells if now < sp["end"]]
        slot_mask, _ = compute_action_masks(card_counts, hero_states, spatial[7])

        def noop(reason: str) -> FactorizedAction:
            return FactorizedAction(0, None, "noop", reason, self.phase, slot_mask, None)

        # Drain queued opening / main-push taps one per tick
        if self.pending_queue:
            slot, cell, reason = self.pending_queue.pop(0)
            _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], slot)
            if slot_mask[slot] and gmask[cell[1], cell[0]]:
                return FactorizedAction(slot, cell, "deploy", reason, self.phase, slot_mask, gmask)
            return noop(f"skipped_illegal_{reason}")

        # State 1: SCOUT_AND_FUNNEL
        if self.phase == "SCOUT_AND_FUNNEL":
            if self.plan is None:
                self._enqueue_opening(spatial, now)
                return self.step(state, card_counts, hero_states, army_groups, enemy_hero_tiles, fallen_hero_tile)
            if now - (self.funnel_started_at or now) >= self.funnel_wait_s:
                self.phase = "MAIN_PUSH"
                self._enqueue_main_push(now)
                return self.step(state, card_counts, hero_states, army_groups, enemy_hero_tiles, fallen_hero_tile)
            return noop("waiting_funnel_timer")

        # State 2: MAIN_PUSH
        if self.phase == "MAIN_PUSH":
            self.phase = "CLOSED_LOOP_MICRO"
            return noop("main_push_complete")

        # State 3 & 4: CLOSED_LOOP_MICRO -> CLEANUP
        destruction = float(state["global"][1])
        alive_defs = state.get("alive_buildings", [])
        is_endgame = (
            (now >= 30.0 and len(alive_defs) <= 5)
            or (now >= 35.0 and destruction >= 0.72)
            or (now >= 65.0)
        )
        if self.enable_cleanup and (destruction >= 0.70 or is_endgame) and card_counts.get(SLOTS["archer"], 0) > 0:
            self.phase = "CLEANUP"

        army_tile = state.get("army_tile")
        army_vel = state.get("army_vel_tile", np.zeros(2))
        primary = army_groups[0] if army_groups else None
        bars = primary.get("bars", []) if primary else []
        yellow_cnt = sum(1 for b in bars if b.get("yellow"))

        # Track short-term bar count history (0.7s window), sustained yellow HP, and hero HUD HP drops (1.6s window)
        if primary and primary.get("source") == "health_bars":
            self.bar_history.append((now, len(bars)))
        self.bar_history = [(ts, cnt) for ts, cnt in self.bar_history if now - ts <= 0.7]
        # A Miner/Hog swarm has ~60 troops; 1-3 injured/red bars on single-target fire is normal!
        # Require >=4 yellow/red bars (>=45% of visible bars) sustained across frames before declaring swarm low-HP.
        low_hp_now = (
            primary is not None
            and primary.get("source") == "health_bars"
            and len(bars) >= 4
            and yellow_cnt >= 4
            and (yellow_cnt / len(bars)) >= 0.45
        )
        sustained_low_hp = (
            low_hp_now
            and (self.yellow_seen_at is not None)
            and ((now - self.yellow_seen_at) <= 0.8)
        )
        self.yellow_seen_at = now if low_hp_now else None
        warden_hp, warden_hp_drop = self._update_hero_hp_drop(
            now, SLOTS["grand_warden"], (hero_hps or {}).get(SLOTS["grand_warden"])
        )
        rc_hp, rc_hp_drop = self._update_hero_hp_drop(
            now, SLOTS["royal_champion"], (hero_hps or {}).get(SLOTS["royal_champion"])
        )

        # Locate the Guardian accurately using continuous shrine-seeded badge tracking (only while alive!)
        guardian = state.get("guardian")
        if guardian is not None and not guardian.alive:
            guardian = None
        guardian_tile = live_guardian_tile if (guardian is not None and live_guardian_tile is not None) else (
            guardian.tile if guardian is not None else None
        )

        # --- 1. Totem Spell (pull ANY Guardian away from main army, or tank heavy core defense if no Guardian) ---
        totem_slot = SLOTS["totem_spell"]
        if (
            self.enable_totem
            and slot_mask[totem_slot]
            and army_tile is not None
            and now >= 4.0
        ):
            totem_anchor = None
            totem_reason = None
            if guardian is not None and guardian_tile is not None:
                dist_g = math.hypot(army_tile[0] - guardian_tile[0], army_tile[1] - guardian_tile[1])
                if (guardian_awake and dist_g <= 11.0) or dist_g <= 8.5:
                    totem_anchor = guardian_tile
                    totem_reason = f"totem_pull_{guardian.name}"
            elif now >= 10.0:
                # Fallback when no Guardian detected: drop Totem behind the closest healthy (>=25% HP) heavy core defense
                heavy = [
                    b for b in state.get("alive_buildings", [])
                    if b.name in ("monolith", "inferno_tower", "firespitter", "scattershot", "revenge_tower")
                    and getattr(b, "hp_frac", 1.0) >= 0.25
                    and math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) <= 8.5
                ]
                if heavy:
                    tgt_b = min(heavy, key=lambda b: math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]))
                    totem_anchor = tgt_b.tile
                    totem_reason = f"totem_tank_{tgt_b.name}"
            if totem_anchor is not None:
                to_army = army_tile - totem_anchor
                norm = float(np.linalg.norm(to_army))
                dir_to_army = to_army / norm if norm > 0.5 else np.array([1.0, 0.0])
                if guardian is not None and guardian.name == "guardian_smasher" and norm > 3.5:
                    # Melee Smasher charges toward the main army during Totem's ~1.0s drop windup:
                    # lead ~2.0 tiles ahead along his chase vector + slight lateral offset so he runs right into it!
                    lead_dist = min(2.2, norm * 0.45)
                    perp = np.array([-dir_to_army[1], dir_to_army[0]])
                    target = np.clip(totem_anchor + lead_dist * dir_to_army + 1.2 * perp, 2.0, GRID_SIZE - 3.0)
                else:
                    # Ranged Guardians (Log Thrower / Longshot) or stationary defenses: drop 1.2 tiles behind them
                    target = np.clip(totem_anchor - 1.2 * dir_to_army, 2.0, GRID_SIZE - 3.0)
                gx, gy = int(round(target[0])), int(round(target[1]))
                _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], totem_slot)
                return FactorizedAction(
                    totem_slot, (gx, gy), "spell", totem_reason, self.phase, slot_mask, gmask
                )

        # --- 1b. Royal Champion Ability (staggered with Warden Tome; freezes nearby defenses for ~7s) ---
        rc_slot = SLOTS["royal_champion"]
        tome_active = now < self.warden_tome_end
        rc_low_or_dropping = (rc_hp is not None) and ((rc_hp <= 0.42) or (rc_hp_drop >= 0.20))
        if self.enable_rc_ability and hero_states[3] == 1 and army_tile is not None and now >= 8.0:
            gx = int(np.clip(round(army_tile[0]), 0, GRID_SIZE - 1))
            gy = int(np.clip(round(army_tile[1]), 0, GRID_SIZE - 1))
            core_threat = float(spatial[1, gy, gx])
            nearby_core = sum(
                1 for b in alive_defs
                if b.is_core and math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) <= 8.0
            )
            # Defer RC ability while Warden Eternal Tome is active unless RC herself is losing HP outside the aura
            if (not tome_active and (core_threat >= 1.0 or nearby_core >= 2 or rc_low_or_dropping)) or (
                tome_active and rc_low_or_dropping
            ):
                self.rc_ability_end = now + 7.0
                reason = "rc_low_hp_save" if rc_low_or_dropping else "rc_heavy_fire"
                return FactorizedAction(rc_slot, None, "hero_ability", reason, self.phase, slot_mask, None)

        # --- 2. Grand Warden Eternal Tome (ONLY when Warden covers main army; staggered after RC freeze!) ---
        warden_slot = SLOTS["grand_warden"]
        warden_tile = state.get("warden_tile")
        pushed_deep = True
        if self.entry_center is not None and army_tile is not None:
            pushed_deep = math.hypot(army_tile[0] - self.entry_center[0], army_tile[1] - self.entry_center[1]) >= self.warden_min_push_tiles
        warden_covers_army = False
        if warden_tile is not None and army_tile is not None:
            # Warden Tome radius is ~8.5 tiles: require main army centroid to be <= 6.8 tiles from Warden
            # so Eternal Tome actually protects the bulk of the army (not popped while Warden lags far behind!).
            warden_covers_army = math.hypot(warden_tile[0] - army_tile[0], warden_tile[1] - army_tile[1]) <= 6.8
        elif now >= 26.0 and army_tile is not None:
            # Late emergency fallback only if Warden aura couldn't be visually locked
            warden_covers_army = True
        if (
            self.enable_warden_ability
            and hero_states[2] == 1
            and army_tile is not None
            and now >= self.warden_min_delay_s
            and pushed_deep
            and warden_covers_army
        ):
            gx = int(np.clip(round(army_tile[0]), 0, GRID_SIZE - 1))
            gy = int(np.clip(round(army_tile[1]), 0, GRID_SIZE - 1))
            core_threat = float(spatial[1, gy, gx])
            if guardian_tile is not None:
                guardian_near = math.hypot(army_tile[0] - guardian_tile[0], army_tile[1] - guardian_tile[1]) <= 7.0
            else:
                guardian_near = any(
                    math.hypot(army_tile[0] - et[0], army_tile[1] - et[1]) <= 6.0
                    for et in (enemy_hero_tiles or [])
                )
            peak_bars = max((cnt for _, cnt in self.bar_history), default=0)
            sharp_drop = peak_bars >= 4 and len(bars) > 0 and len(bars) <= peak_bars * 0.65
            # Real HP-loss override during RC's 7s freeze: NEVER trigger on bar-count dip alone (Miners burrow
            # underground and healed troops hide full-HP bars!). Require sustained yellow/red troop damage OR Warden HP drop!
            real_hp_dropping = (
                (sustained_low_hp and yellow_cnt >= 4 and (yellow_cnt / max(1, len(bars))) >= 0.45)
                or ((warden_hp is not None) and (warden_hp <= 0.45 or warden_hp_drop >= 0.22))
            )
            rc_freeze_active = now < self.rc_ability_end
            if rc_freeze_active:
                # Defenses are already frozen by RC: hold Eternal Tome UNLESS army/Warden is genuinely losing HP!
                if real_hp_dropping:
                    self.warden_tome_end = now + 7.5
                    return FactorizedAction(
                        warden_slot, None, "hero_ability", "override_rc_freeze_hp_drop", self.phase, slot_mask, None
                    )
            elif (
                (guardian_near and (yellow_cnt >= 1 or sharp_drop or len(bars) >= 3))
                or (core_threat >= 0.90 and (sharp_drop or yellow_cnt >= 2 or (now >= 18.0 and len(bars) >= 2)))
                or (now >= 26.0 and len(bars) >= 2)
                or real_hp_dropping
            ):
                self.warden_tome_end = now + 7.5
                if guardian_near:
                    reason = f"guardian_{guardian.name}" if guardian is not None else "guardian_engagement"
                elif sharp_drop or real_hp_dropping:
                    reason = "burst_bar_drop"
                elif core_threat >= 0.90:
                    reason = "core_heavy_fire"
                else:
                    reason = "core_timely_tome"
                return FactorizedAction(warden_slot, None, "hero_ability", reason, self.phase, slot_mask, None)

        # --- 3. Closed-loop Healing Spell (unfrozen splash-threat + real damage check; tight lead on army!) ---
        heal_slot = SLOTS["healing_spell"]
        heal_cd = now - self.last_heal_at
        if (
            slot_mask[heal_slot]
            and army_tile is not None
            and primary is not None
            and now >= self.warden_tome_end
            and heal_cd >= 3.5
        ):
            if primary.get("source") == "health_bars" and len(bars) >= 2:
                alive = alive_defs
                freeze_slot_id = SLOTS["freeze_spell"]
                # Exclude defenses currently frozen by an active Freeze Spell from splash_score!
                unfrozen_alive = [
                    b for b in alive
                    if not any(
                        sp["slot"] == freeze_slot_id
                        and math.hypot(b.tile[0] - sp["tile"][0], b.tile[1] - sp["tile"][1]) <= 3.6
                        for sp in self.active_spells
                    )
                ]
                splash_score = 0.0
                for b in unfrozen_alive:
                    w = HEAL_SPLASH_WEIGHTS.get(b.name, 0.0)
                    if w <= 0.0:
                        continue
                    max_r = b.outer_range - 0.3 if b.facing_rad is not None else min(b.outer_range - 0.3, 10.5)
                    if b.covers_tile(army_tile, max_range=max_r, margin=0.3):
                        # Discount nearly-dead defenses (e.g. 10% HP Firespitter about to die in <1s!)
                        hp_f = min(1.0, max(0.0, float(getattr(b, "hp_frac", 1.0))) / 0.35)
                        splash_score += w * hp_f

                # Only count HEALTHY (hp_frac >= 0.30) unfrozen defenses as stay anchors:
                # if the troops are only finishing off a sliver-HP building, they will immediately move on!
                stay_defs = [
                    b for b in unfrozen_alive
                    if float(getattr(b, "hp_frac", 1.0)) >= 0.30
                    and math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) <= 6.0
                ]
                has_stay_pocket = len(stay_defs) >= 2 or splash_score >= 2.2
                # Even if current target is dying (<30% HP), allow critical yellow-HP Heal if a healthy next target is nearby!
                next_healthy_defs = [
                    b for b in alive
                    if float(getattr(b, "hp_frac", 1.0)) >= 0.35
                    and math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) <= 8.5
                ]
                # For a ~60-troop Miner/Hog swarm, 2-3 red/yellow bars without splash is just single-target attrition.
                # Under real unfrozen splash (>=2.0): trigger on sustained >=4 yellow/red bars (>=45%).
                # Without splash (<2.0, e.g. Giant Bomb): require sustained >=6 yellow/red bars (>=50%)!
                critical_hp = sustained_low_hp and yellow_cnt >= 6 and (yellow_cnt / max(1, len(bars))) >= 0.50
                hp_trigger = (
                    sustained_low_hp
                    and (
                        (has_stay_pocket and splash_score >= 2.0)
                        or (critical_hp and (has_stay_pocket or len(next_healthy_defs) >= 1))
                    )
                )

                # Heal placement with dying-building lookahead:
                # Exclude sliver-HP (<30%) defenses from pocket centroid; if current target is nearly dead,
                # lead Heal toward the NEXT healthy compartment (<=8.5t) so troops walk right into the Heal!
                lead = np.zeros(2, dtype=float)
                healthy_close = [
                    b for b in alive
                    if float(getattr(b, "hp_frac", 1.0)) >= 0.30
                    and math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) <= 5.5
                ]
                if healthy_close:
                    max_lead = 1.5 if hp_trigger else 1.7
                    lead_frac = 0.50 if hp_trigger else 0.58
                    pocket = healthy_close
                elif next_healthy_defs:
                    # Current defense is dying (<30% HP) -> look ahead toward the next healthy compartment!
                    max_lead = 2.4
                    lead_frac = 0.62
                    pocket = next_healthy_defs
                else:
                    max_lead = 1.4
                    lead_frac = 0.45
                    pocket = []

                if pocket:
                    weights = np.array([
                        float(getattr(b, "hp_frac", 1.0))
                        * (2.5 if b.is_splash else (1.8 if b.is_core else 0.8))
                        / max(1.5, math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]))
                        for b in pocket
                    ], dtype=float)
                    pocket_center = sum(w * b.tile for w, b in zip(weights, pocket)) / float(weights.sum())
                    vec = pocket_center - army_tile
                    d_next = float(np.linalg.norm(vec))
                    if d_next > 0.5:
                        lead = vec * (min(max_lead, d_next * lead_frac) / d_next)
                elif float(np.linalg.norm(army_vel)) > 0.3:
                    speed = float(np.linalg.norm(army_vel))
                    lead = army_vel * min(1.2, max_lead / max(speed, 1e-3))
                target_tile = np.clip(army_tile + lead, 0.0, GRID_SIZE - 1.0)
                gx, gy = int(round(target_tile[0])), int(round(target_tile[1]))

                already_healing = any(
                    sp["slot"] == heal_slot and math.hypot(army_tile[0] - sp["tile"][0], army_tile[1] - sp["tile"][1]) < (HEAL_RADIUS + 1.5)
                    for sp in self.active_spells
                )
                # Never waste a preemptive Heal on 100% healthy full-green troops (especially during RC freeze!):
                # require at least 1 damaged/yellow troop OR genuinely extreme unfrozen multi-splash deep in the core.
                rc_freeze_now = now < self.rc_ability_end
                in_splash = (
                    not rc_freeze_now
                    and heal_cd >= 6.5
                    and len(bars) >= 4
                    and has_stay_pocket
                    and (
                        (splash_score >= 3.8 and yellow_cnt >= 3)
                        or (splash_score >= 5.0 and pushed_deep and len(bars) >= 5)
                    )
                )
                if not already_healing and (hp_trigger or in_splash):
                    reason = "yellow_health" if hp_trigger else f"splash_score_{splash_score:.1f}"
                    _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], heal_slot)
                    self.record_spell(now, heal_slot, (gx, gy))
                    self.last_heal_at = now
                    return FactorizedAction(heal_slot, (gx, gy), "spell", reason, self.phase, slot_mask, gmask)

        # --- 3. Closed-loop Freeze Spell on high-threat core defense targeting main army ---
        freeze_slot = SLOTS["freeze_spell"]
        if self.enable_freeze and slot_mask[freeze_slot] and army_tile is not None:
            best_target = None
            best_prio = 0.0
            for b in state.get("alive_buildings", []):
                prio = FREEZE_PRIORITY_DEFENSES.get(b.name, 0.0)
                if prio <= 0 or float(getattr(b, "hp_frac", 1.0)) < 0.25:
                    continue
                dist = math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1])
                eff_range = b.outer_range if b.outer_range > 0 else 8.0
                max_r = eff_range if b.facing_rad is not None else min(eff_range, 9.5)
                if not b.covers_tile(army_tile, max_range=max_r):
                    continue
                already_frozen = any(
                    sp["slot"] == freeze_slot and math.hypot(b.tile[0] - sp["tile"][0], b.tile[1] - sp["tile"][1]) < 3.2
                    for sp in self.active_spells
                )
                if already_frozen:
                    continue
                score = prio - 0.15 * dist
                if score > best_prio:
                    best_prio = score
                    best_target = b
            if best_target is not None:
                gx = int(np.clip(round(best_target.tile[0]), 0, GRID_SIZE - 1))
                gy = int(np.clip(round(best_target.tile[1]), 0, GRID_SIZE - 1))
                _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], freeze_slot)
                self.record_spell(now, freeze_slot, (gx, gy))
                return FactorizedAction(
                    freeze_slot, (gx, gy), "spell", f"freeze_{best_target.name}", self.phase, slot_mask, gmask
                )

        # --- 3c. Poison Spell on Guardian or defending enemy heroes engaging the army ---
        poison_slot = SLOTS["poison_spell"]
        if slot_mask[poison_slot] and army_tile is not None and now >= 4.0:
            poison_target = None
            poison_reason = None
            if guardian is not None and guardian_tile is not None:
                if math.hypot(army_tile[0] - guardian_tile[0], army_tile[1] - guardian_tile[1]) <= 9.5:
                    poison_target = guardian_tile
                    poison_reason = f"poison_{guardian.name}"
            elif enemy_hero_tiles:
                near_enemy = min(
                    enemy_hero_tiles,
                    key=lambda et: math.hypot(army_tile[0] - et[0], army_tile[1] - et[1]),
                )
                if math.hypot(army_tile[0] - near_enemy[0], army_tile[1] - near_enemy[1]) <= 8.5:
                    poison_target = near_enemy
                    poison_reason = "poison_enemy_hero"
            if poison_target is not None:
                gx = int(np.clip(round(poison_target[0]), 0, GRID_SIZE - 1))
                gy = int(np.clip(round(poison_target[1]), 0, GRID_SIZE - 1))
                _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], poison_slot)
                return FactorizedAction(poison_slot, (gx, gy), "spell", poison_reason, self.phase, slot_mask, gmask)

        # --- 4. Optional Revive Spell (via HUD hero portrait tap for 100% accuracy, or map tile fallback) ---
        revive_slot = SLOTS["revive_spell"]
        if self.enable_revive and slot_mask[revive_slot] and fallen_hero_tile is not None:
            if isinstance(fallen_hero_tile, int):
                # HUD-card revive: select Revive Spell then tap the fallen hero's HUD portrait directly!
                return FactorizedAction(
                    revive_slot, (fallen_hero_tile, -1), "revive_hud", f"revive_hero_slot_{fallen_hero_tile}",
                    self.phase, slot_mask, None,
                )
            gx = int(np.clip(round(fallen_hero_tile[0]), 0, GRID_SIZE - 1))
            gy = int(np.clip(round(fallen_hero_tile[1]), 0, GRID_SIZE - 1))
            _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], revive_slot)
            return FactorizedAction(revive_slot, (gx, gy), "spell", "revive_hero", self.phase, slot_mask, gmask)

        # --- 5. Optional Cleanup (includes zero-label non-defense trash buildings!) ---
        if self.phase == "CLEANUP" and slot_mask[SLOTS["archer"]] and army_tile is not None:
            alive = [*alive_defs, *state.get("alive_non_defense", [])]
            if alive:
                farthest = max(alive, key=lambda b: math.hypot(b.tile[0] - army_tile[0], b.tile[1] - army_tile[1]))
                deploy_rows, deploy_cols = np.where(spatial[7] >= 0.5)
                if len(deploy_rows) > 0:
                    idx = int(np.argmin((deploy_cols - farthest.tile[0]) ** 2 + (deploy_rows - farthest.tile[1]) ** 2))
                    cell = (int(deploy_cols[idx]), int(deploy_rows[idx]))
                    _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], SLOTS["archer"])
                    return FactorizedAction(SLOTS["archer"], cell, "deploy", "cleanup_farthest", self.phase, slot_mask, gmask)

        # --- 6. Endgame Value Maximization: spend remaining beneficial spells & hero abilities before battle ends ---
        # Note: Ice Block (slot 14) is intentionally excluded because freezing healthy cleanup troops reduces DPS.
        if is_endgame and (now - self.last_endgame_dump_at) >= 1.2:
            # 6a. Leftover Healing Spell on active army approaching remaining defenses (or low-HP cleanup)
            already_healing_end = army_tile is not None and any(
                sp["slot"] == heal_slot and math.hypot(army_tile[0] - sp["tile"][0], army_tile[1] - sp["tile"][1]) < (HEAL_RADIUS + 1.5)
                for sp in self.active_spells
            )
            if (
                slot_mask[heal_slot]
                and army_tile is not None
                and now >= self.warden_tome_end
                and heal_cd >= 4.0
                and not already_healing_end
                and (len(alive_defs) >= 1 or yellow_cnt >= 1 or now >= 70.0)
            ):
                target_tile = army_tile.copy()
                if alive_defs:
                    nearest_d = min(alive_defs, key=lambda b: math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]))
                    vec = nearest_d.tile - army_tile
                    dn = float(np.linalg.norm(vec))
                    if dn > 0.5:
                        target_tile = np.clip(army_tile + vec * (min(2.0, dn * 0.6) / dn), 0.0, GRID_SIZE - 1.0)
                gx, gy = int(round(target_tile[0])), int(round(target_tile[1]))
                _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], heal_slot)
                self.record_spell(now, heal_slot, (gx, gy))
                self.last_heal_at = now
                self.last_endgame_dump_at = now
                return FactorizedAction(heal_slot, (gx, gy), "spell", "endgame_heal", self.phase, slot_mask, gmask)

            # 6b. Leftover Freeze Spell on any highest-value remaining defense, Guardian, or enemy hero
            if self.enable_freeze and slot_mask[freeze_slot]:
                best_tile = None
                best_label = None
                best_score = -1e9
                for b in alive_defs:
                    if any(
                        sp["slot"] == freeze_slot and math.hypot(b.tile[0] - sp["tile"][0], b.tile[1] - sp["tile"][1]) < 3.2
                        for sp in self.active_spells
                    ):
                        continue
                    dist = math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) if army_tile is not None else 0.0
                    base = FREEZE_PRIORITY_DEFENSES.get(b.name, 1.5 if b.is_core else (1.0 if b.is_splash else 0.6))
                    sc = base - 0.08 * dist
                    if sc > best_score:
                        best_score = sc
                        best_tile = b.tile
                        best_label = b.name
                if best_tile is None and guardian is not None and guardian_tile is not None:
                    best_tile, best_label = guardian_tile, guardian.name
                elif best_tile is None and enemy_hero_tiles:
                    best_tile, best_label = enemy_hero_tiles[0], "enemy_hero"
                if best_tile is not None:
                    gx = int(np.clip(round(best_tile[0]), 0, GRID_SIZE - 1))
                    gy = int(np.clip(round(best_tile[1]), 0, GRID_SIZE - 1))
                    _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], freeze_slot)
                    self.record_spell(now, freeze_slot, (gx, gy))
                    self.last_endgame_dump_at = now
                    return FactorizedAction(
                        freeze_slot, (gx, gy), "spell", f"endgame_freeze_{best_label}", self.phase, slot_mask, gmask
                    )

            # 6c. Leftover Totem Spell to tank/distract remaining defense, Guardian, or nearest building
            if self.enable_totem and slot_mask[totem_slot]:
                totem_pt = None
                totem_lbl = None
                if guardian is not None and guardian_tile is not None:
                    totem_pt, totem_lbl = guardian_tile, guardian.name
                elif alive_defs:
                    tgt_b = min(
                        alive_defs,
                        key=lambda b: (
                            -(2.0 if b.is_core else (1.0 if b.is_splash else 0.0)),
                            math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) if army_tile is not None else 0.0,
                        ),
                    )
                    totem_pt, totem_lbl = tgt_b.tile, tgt_b.name
                elif state.get("alive_non_defense"):
                    nb = min(
                        state["alive_non_defense"],
                        key=lambda b: math.hypot(army_tile[0] - b.tile[0], army_tile[1] - b.tile[1]) if army_tile is not None else 0.0,
                    )
                    totem_pt, totem_lbl = nb.tile, "cleanup_building"
                if totem_pt is not None:
                    gx = int(np.clip(round(totem_pt[0]), 2, GRID_SIZE - 3))
                    gy = int(np.clip(round(totem_pt[1]), 2, GRID_SIZE - 3))
                    _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], totem_slot)
                    self.last_endgame_dump_at = now
                    return FactorizedAction(
                        totem_slot, (gx, gy), "spell", f"endgame_totem_{totem_lbl}", self.phase, slot_mask, gmask
                    )

            # 6d. Leftover Poison Spell on Guardian / enemy hero / army engagement zone
            if slot_mask[poison_slot]:
                p_pt = None
                p_lbl = None
                if guardian is not None and guardian_tile is not None:
                    p_pt, p_lbl = guardian_tile, guardian.name
                elif enemy_hero_tiles:
                    p_pt, p_lbl = enemy_hero_tiles[0], "enemy_hero"
                elif army_tile is not None and (len(alive_defs) >= 1 or now >= 65.0):
                    p_pt, p_lbl = army_tile, "army_zone"
                if p_pt is not None:
                    gx = int(np.clip(round(p_pt[0]), 0, GRID_SIZE - 1))
                    gy = int(np.clip(round(p_pt[1]), 0, GRID_SIZE - 1))
                    _, gmask = compute_action_masks(card_counts, hero_states, spatial[7], poison_slot)
                    self.last_endgame_dump_at = now
                    return FactorizedAction(
                        poison_slot, (gx, gy), "spell", f"endgame_poison_{p_lbl}", self.phase, slot_mask, gmask
                    )

            # 6e. Unspent RC / Warden abilities near endgame (still respecting stagger!)
            if self.enable_rc_ability and hero_states[3] == 1 and now >= self.warden_tome_end and (len(alive_defs) >= 1 or now >= 55.0):
                self.rc_ability_end = now + 7.0
                self.last_endgame_dump_at = now
                return FactorizedAction(rc_slot, None, "hero_ability", "endgame_rc_ability", self.phase, slot_mask, None)

            if self.enable_warden_ability and hero_states[2] == 1 and now >= self.rc_ability_end and (len(alive_defs) >= 1 or now >= 60.0):
                self.warden_tome_end = now + 7.5
                self.last_endgame_dump_at = now
                return FactorizedAction(warden_slot, None, "hero_ability", "endgame_warden_tome", self.phase, slot_mask, None)

        return noop("monitoring")
