"""Find a narrow deployable grass ring outside the visible red base boundary."""

import math

import cv2
import numpy as np

from grid_projection import GRID_SIZE


def red_boundary(image, projection):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    red, green, blue = cv2.split(image.astype(np.int16))
    mask = (
        ((hue < 22) | (hue > 169)) & (saturation > 85) & (value > 85)
        & (red - green >= 25) & (red - blue >= 25)
    ).astype(np.uint8) * 255
    mask[:70] = 0
    mask[665:] = 0
    segments = cv2.HoughLinesP(mask, 1, np.pi/360, 26, minLineLength=100, maxLineGap=12)
    edges = {}
    if segments is None:
        return None, hsv
    for sx, sy, ex, ey in segments.reshape(-1, 4):
        start = projection.pixel_to_grid(sx, sy)
        end = projection.pixel_to_grid(ex, ey)
        difference = [abs(b-a) for a,b in zip(start,end)]
        axis = int(difference[1] < difference[0])
        if difference[axis] > 1.5 or difference[1-axis] < 10:
            continue
        coordinate = (start[axis] + end[axis]) / 2
        side = int(coordinate > GRID_SIZE/2)
        if abs(coordinate - side*GRID_SIZE) > 4:
            continue
        length = math.hypot(ex-sx, ey-sy)
        key = (axis, side)
        if key not in edges or length > edges[key][0]:
            edges[key] = (length, coordinate)

    if len(edges) != 4:
        return None, hsv
    bounds = tuple(edges[axis, side][1] for axis in range(2) for side in range(2))
    left, right, top, bottom = bounds
    if not (35 < right-left < 47 and 35 < bottom-top < 47):
        return None, hsv
    return bounds, hsv


def outer_lawn_cells(image, projection, *, padding=0):
    """Return grid cells and red-boundary coordinates in the normalized game frame."""
    boundary, hsv = red_boundary(np.asarray(image.convert("RGB")), projection)
    if boundary is None:
        return [], None
    left, right, top, bottom = boundary
    hue, saturation, value = cv2.split(hsv)
    grass = (hue >= 31) & (hue <= 59) & (saturation >= 85) & (value >= 112)
    cells = []
    height, width = grass.shape
    for row in range(-padding, GRID_SIZE + padding):
        for column in range(-padding, GRID_SIZE + padding):
            c, r = column+.5, row+.5
            outside = math.hypot(max(left-c, c-right, 0), max(top-r, r-bottom, 0))
            if not .70 <= outside <= 2.8:
                continue
            x, y = map(round, projection.grid_to_pixel(c, r))
            if (12 <= x < width-12 and 75 <= y < height*.82
                    and grass[y-5:y+6, x-5:x+6].mean() >= .88):
                cells.append((column, row))
    return cells, boundary


def apply_outer_lawn(state, cells):
    """Replace any deployment polygons with an automatically derived mask."""
    channels = state["channels"]
    known = channels.index("deployment_known")
    allowed = channels.index("deployment_allowed")
    selected = set(cells)
    for row, columns in enumerate(state["tensor"]):
        for column, cell in enumerate(columns):
            cell[known] = cell[allowed] = int((column, row) in selected)
