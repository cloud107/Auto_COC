"""Calibrate a diamond grid and inspect its alignment on a game screenshot."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

GRID_SIZE = 44


@dataclass(frozen=True)
class GridProjection:
    """Affine projection from continuous grid coordinates to viewport pixels."""

    origin: tuple[float, float]
    column_step: tuple[float, float]
    row_step: tuple[float, float]

    @classmethod
    def from_corners(cls, origin, column_end, row_end):
        column_step = tuple((end - start) / GRID_SIZE for start, end in zip(origin, column_end))
        row_step = tuple((end - start) / GRID_SIZE for start, end in zip(origin, row_end))
        projection = cls(origin, column_step, row_step)
        if abs(projection.determinant) < 1e-8:
            raise ValueError("Grid calibration corners must be non-collinear")
        return projection

    @property
    def determinant(self):
        ax, ay = self.column_step
        bx, by = self.row_step
        return ax * by - bx * ay

    def grid_to_pixel(self, column, row):
        x, y = self.origin
        ax, ay = self.column_step
        bx, by = self.row_step
        return x + column * ax + row * bx, y + column * ay + row * by

    def pixel_to_grid(self, x, y):
        x -= self.origin[0]
        y -= self.origin[1]
        ax, ay = self.column_step
        bx, by = self.row_step
        det = self.determinant
        return (by * x - bx * y) / det, (ax * y - ay * x) / det


def projection_for_viewport(calibration, width, height):
    if abs(width / height - calibration["viewport_aspect"]) > 0.015:
        raise ValueError("Screenshot aspect ratio differs from the calibrated viewport")
    corners = []
    for name in ("origin", "column_end", "row_end"):
        x, y = calibration[name]
        corners.append((x * width, y * height))
    return GridProjection.from_corners(*corners)


def draw_grid(image, projection):
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(overlay)
    for i in range(GRID_SIZE + 1):
        major = i % 11 == 0
        color = (52, 216, 255, 220 if major else 90)
        width = 3 if major else 1
        draw.line([projection.grid_to_pixel(i, 0), projection.grid_to_pixel(i, GRID_SIZE)],
                  fill=color, width=width)
        draw.line([projection.grid_to_pixel(0, i), projection.grid_to_pixel(GRID_SIZE, i)],
                  fill=color, width=width)
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("calibration", type=Path, help="JSON with normalized viewport corner coordinates")
    parser.add_argument("output", type=Path, help="PNG path for the grid overlay")
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    with Image.open(args.screenshot) as screenshot:
        try:
            projection = projection_for_viewport(calibration, screenshot.width, screenshot.height)
        except ValueError as error:
            parser.error(str(error))
        draw_grid(screenshot, projection).save(args.output)


if __name__ == "__main__":
    main()
