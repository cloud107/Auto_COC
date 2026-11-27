"""Prepare an incremental YOLO labeling workspace preserving all 24 existing defense labels.

Copies images + existing 24-class defense/TH/Guardian YOLO .txt labels from
datasets/realistic-defense24-self-20261125/ into a new workspace and writes
classes.txt + data.yaml with fine-grained non-defense classes and the 6 Hero Flag classes
(flag_bk, flag_aq, flag_gw, flag_rc, flag_mp, flag_dd) appended cleanly as IDs 24..46.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXISTING_24_CLASSES = [
    "town_hall",                 # 0
    "inferno_tower",             # 1
    "scattershot",               # 2
    "monolith",                  # 3
    "spell_tower",               # 4
    "x_bow",                     # 5
    "air_defense",               # 6
    "ricochet_cannon",           # 7
    "multi_archer_tower",        # 8
    "firespitter",               # 9
    "revenge_tower",             # 10
    "spiky_tower",               # 11
    "firecracker_tower",         # 12
    "air_sweeper",               # 13
    "multi_gear_tower",          # 14
    "mortar",                    # 15
    "bomb_tower",                # 16
    "clan_castle",               # 17
    "builder_hut",               # 18
    "super_wizard_tower",        # 19
    "guardian_smasher",          # 20
    "guardian_longshot",         # 21
    "guardian_log_thrower",      # 22
    "multi_gear_tower_variant",  # 23
]

# Fine-grained non-defense buildings (for clean YOLO precision; later mapped to tank/trash channels)
# + 6 Defending Hero Flags (skin-invariant & angle-invariant!)
NEW_CLASSES = [
    # High-HP non-defense tanks (24..28)
    "gold_storage",              # 24
    "elixir_storage",            # 25
    "dark_elixir_storage",       # 26
    "hero_hall",                 # 27
    "blacksmith",                # 28
    # Standard / trash non-defense buildings (29..40)
    "gold_mine",                 # 29
    "elixir_collector",          # 30
    "dark_elixir_drill",         # 31
    "army_camp",                 # 32
    "barracks",                  # 33
    "dark_barracks",             # 34
    "laboratory",                # 35
    "spell_factory",             # 36
    "dark_spell_factory",        # 37
    "siege_workshop",            # 38
    "pet_house",                 # 39
    "helper_hut",                # 40
    # 6 Defending Hero Flags (41..46) - label the ground flag/banner, NOT hero body!
    "flag_bk",                   # 41 Barbarian King banner
    "flag_aq",                   # 42 Archer Queen banner
    "flag_gw",                   # 43 Grand Warden banner
    "flag_rc",                   # 44 Royal Champion banner
    "flag_mp",                   # 45 Minion Prince banner
    "flag_dd",                   # 46 Dragon Duke banner
]

ALL_CLASSES = EXISTING_24_CLASSES + NEW_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=ROOT / "datasets" / "realistic-defense24-self-20261125",
        help="Source 24-class YOLO dataset",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=ROOT / "datasets" / "prebattle-full-v1",
        help="Destination incremental labeling workspace",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source dataset not found: {args.src}")
    if args.dst.exists() and not args.overwrite:
        raise SystemExit(f"Destination already exists: {args.dst} (pass --overwrite to rebuild)")

    counts = Counter()
    copied = 0
    for split in ("train", "val", "test"):
        img_src = args.src / "images" / split
        lbl_src = args.src / "labels" / split
        if not img_src.exists():
            continue
        img_dst = args.dst / "images" / split
        lbl_dst = args.dst / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(img_src.glob("*.*")):
            shutil.copy2(img_path, img_dst / img_path.name)
            lbl_path = lbl_src / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        counts[int(line.split()[0])] += 1
            copied += 1

    # Write classes.txt (LabelImg / X-AnyLabeling compatible) inside root and each labels/<split>
    classes_txt = "\n".join(ALL_CLASSES) + "\n"
    (args.dst / "classes.txt").write_text(classes_txt, encoding="utf-8")
    for split in ("train", "val", "test"):
        d = args.dst / "labels" / split
        if d.exists():
            (d / "classes.txt").write_text(classes_txt, encoding="utf-8")

    # Write YOLO data.yaml
    yaml_lines = [
        f"path: {args.dst.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(ALL_CLASSES)],
        "",
    ]
    (args.dst / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    print(f"Prepared incremental labeling kit at {args.dst}")
    print(f"Copied {copied} images + existing 24-class labels ({sum(counts.values())} preserved boxes).")
    print("Classes needing boost (count < 30 among existing 24):")
    for cid in range(len(EXISTING_24_CLASSES)):
        if counts[cid] < 30:
            print(f"  {cid:2d}: {EXISTING_24_CLASSES[cid]:25s} ({counts[cid]} boxes)")
    print(f"Ready to add {len(NEW_CLASSES)} new classes (IDs 24..{len(ALL_CLASSES)-1}) in classes.txt!")


if __name__ == "__main__":
    main()
