"""Recognize the calibrated MuMu deployment HUD and match the reference lineup."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

NORMALIZED_SIZE = (1000, 800)
CAPTURES = Path(__file__).parent / "captures"
REFERENCE = CAPTURES / "self-attack-current-army-predeploy.png"
BATTLE_REFERENCE = CAPTURES / "self-attack-battle-hud.png"
RESULT_REFERENCE = CAPTURES / "self-attack-result-hud.png"
DEPLOYMENT_MARKERS = ((30, 675, 128, 699), (449, 16, 551, 32))
BATTLE_MARKERS = ((25, 673, 135, 700), (434, 15, 566, 34), (934, 553, 978, 594))
RESULT_MARKERS = ((435, 641, 570, 686), (244, 318, 278, 360), (727, 324, 757, 364))
CARD_CENTERS = (107, 157, 208, 259, 310, 374, 430, 480, 531, 581, 642, 692, 742, 792, 842, 892)
MAX_MARKER_ERROR = 32
MAX_CARD_ERROR = 34
QUANTITY_SLOTS = frozenset((1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16))
BRIGHT_PIXEL = [0] * 185 + [255] * 71
COUNT_BITS = getattr(int, "bit_count", lambda value: bin(value).count("1"))


def normalized_image(path):
    with Image.open(path) as source:
        width, height = source.size
        if abs(width / height - 1.25) > 0.015:
            raise ValueError(f"Expected a calibrated 5:4 game viewport, got {width}×{height}")
        image = source.convert("RGB").resize(NORMALIZED_SIZE, Image.Resampling.BILINEAR)
    return image, (width, height)


def patch_error(actual, template):
    return sum(ImageStat.Stat(ImageChops.difference(actual, template)).mean) / 3


def quantity_mask(image, center, dx=0, dy=0):
    region = image.crop((center + 5 + dx, 732 + dy, center + 23 + dx, 744 + dy))
    red, green, blue = region.split()
    minimum = ImageChops.darker(ImageChops.darker(red, green), blue)
    return int.from_bytes(minimum.point(BRIGHT_PIXEL).tobytes(), "big")


def quantity_templates(reference, battle_reference):
    return {
        1: quantity_mask(reference, CARD_CENTERS[11]),
        2: quantity_mask(reference, CARD_CENTERS[1]),
        3: quantity_mask(battle_reference, CARD_CENTERS[0], dx=2),
        4: quantity_mask(reference, CARD_CENTERS[0]),
    }


def read_quantity(image, center, templates):
    errors = dict.fromkeys(templates, float("inf"))
    for dx in range(-2, 4):
        for dy in range(-2, 3):
            mask = quantity_mask(image, center, dx, dy)
            for quantity, template in templates.items():
                error = COUNT_BITS(mask ^ template) / (18 * 12 * 8)
                errors[quantity] = min(error, errors[quantity])
    best, second, *_ = sorted(errors, key=errors.get)
    if errors[best] <= 0.10 and errors[second] - errors[best] >= 0.02:
        return best
    return None


def recognize(image, size, reference, battle_reference, result_reference):
    candidates = (
        ("pre_deployment", reference, DEPLOYMENT_MARKERS),
        ("battle", battle_reference, BATTLE_MARKERS),
        ("result", result_reference, RESULT_MARKERS),
    )
    marker_errors = {
        phase: [round(patch_error(image.crop(box), template.crop(box)), 2) for box in boxes]
        for phase, template, boxes in candidates
    }
    matching_phases = [
        phase for phase, errors in marker_errors.items() if max(errors) <= MAX_MARKER_ERROR
    ]
    phase = min(matching_phases, key=lambda name: sum(marker_errors[name]), default="unknown")
    result = {
        "phase": phase,
        "viewport": {"width": size[0], "height": size[1]},
        "marker_errors": marker_errors,
    }
    if phase in ("pre_deployment", "battle"):
        width, height = size
        cards = []
        quantities = quantity_templates(reference, battle_reference)
        for index, x in enumerate(CARD_CENTERS, start=1):
            box = (x - 13, 746, x + 13, 770)
            error = patch_error(image.crop(box), reference.crop(box))
            matches = error <= MAX_CARD_ERROR
            remaining = None
            if matches and index in QUANTITY_SLOTS:
                remaining = read_quantity(image, x, quantities)
            cards.append({
                "slot": index,
                "center": [round(x * width / 1000), round(760 * height / 800)],
                "matches_reference": matches,
                "remaining": remaining,
                "appearance_error": round(error, 2),
            })
        result["reference_cards_matched"] = sum(card["matches_reference"] for card in cards)
        result["cards"] = cards
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    image, size = normalized_image(args.screenshot)
    reference, _ = normalized_image(args.reference)
    battle_reference, _ = normalized_image(BATTLE_REFERENCE)
    result_reference, _ = normalized_image(RESULT_REFERENCE)
    print(json.dumps(recognize(image, size, reference, battle_reference, result_reference), indent=2))


if __name__ == "__main__":
    main()
