# SPDX-License-Identifier: MIT
import numpy as np
import shapely

ASCENDER_SHARE = 0.8
HEIGHT_PERCENTILE = 85
HEIGHT_SAMPLES = 32


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def has_area(points: np.ndarray) -> bool:
    return len(points) >= 3 and shapely.MultiPoint(points).convex_hull.area > 0


def without_repeats(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.concatenate([[True], np.any(np.diff(points, axis=0) != 0, axis=1)])
    return points[keep]


def sample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    positions = (np.arange(count) + 0.5) / count * distances[-1]
    return np.stack([np.interp(positions, distances, points[:, axis]) for axis in (0, 1)], axis=1)


def _reading_direction(polygon: np.ndarray) -> np.ndarray:
    corners = np.asarray(shapely.minimum_rotated_rectangle(shapely.Polygon(polygon)).exterior.coords)
    sides = np.diff(corners[:3], axis=0)
    lengths = np.linalg.norm(sides, axis=1)
    if not lengths.all():
        return np.array([1.0, 0.0])
    sides = sides / lengths[:, np.newaxis]
    vertical = lengths.max() > 3 * lengths.min() and abs(sides[np.argmax(lengths), 1]) > np.sqrt(0.5)
    axis = 1 if vertical else 0
    direction = sides[np.argmax(np.abs(sides[:, axis]))]
    return direction if direction[axis] > 0 else -direction


def baseline_from_polygon(polygon: np.ndarray) -> np.ndarray:
    direction = _reading_direction(polygon)
    up = np.array([direction[1], -direction[0]])
    along = polygon @ direction
    across = polygon @ up
    offset = across.min() + (1 - ASCENDER_SHARE) * (across.max() - across.min())
    return np.stack([position * direction + offset * up for position in (along.min(), along.max())])


def _interval_at_baseline(piece: shapely.Geometry, point: np.ndarray, up: np.ndarray) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    best_distance = np.inf
    for part in shapely.get_parts(piece):
        positions = (shapely.get_coordinates(part) - point) @ up
        if len(positions) < 2:
            continue
        low, high = float(positions.min()), float(positions.max())
        distance = max(low, -high, 0.0)
        if distance < best_distance:
            best, best_distance = (low, high), distance
    return best


def line_heights(baseline: np.ndarray, polygon: np.ndarray) -> tuple[float, float]:
    direction = baseline[-1] - baseline[0]
    length = np.linalg.norm(direction)
    direction = direction / length if length else np.array([1.0, 0.0])
    up = np.array([direction[1], -direction[0]])
    reach = float(np.linalg.norm(np.ptp(polygon, axis=0))) + 1.0

    points = sample_polyline(baseline, HEIGHT_SAMPLES)
    shape = shapely.make_valid(shapely.Polygon(polygon))
    normals = shapely.linestrings(np.stack([points - up * reach, points + up * reach], axis=1))
    pieces = shapely.intersection(shape, normals)
    intervals = [
        interval
        for piece, point in zip(pieces, points, strict=True)
        if (interval := _interval_at_baseline(piece, point, up)) is not None
    ]
    if intervals:
        lows, highs = np.asarray(intervals).T
        ascender = float(np.percentile(highs, HEIGHT_PERCENTILE))
        descender = float(np.percentile(-lows, HEIGHT_PERCENTILE))
        if ascender > 0 and ascender + max(descender, 0.0) >= 1:
            return max(ascender, 1.0), max(descender, 0.0)
    height = float(np.ptp(polygon @ up))
    return max(height * ASCENDER_SHARE, 1.0), height * (1 - ASCENDER_SHARE)


def clip_to_page(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    rounded = np.rint(points).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, max(width - 1, 0))
    rounded[:, 1] = np.clip(rounded[:, 1], 0, max(height - 1, 0))
    return without_repeats(rounded)


def page_polygon(points: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    polygon = clip_to_page(points, size)
    if len(polygon) > 1 and np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    if len(polygon) < 3 or shapely.Polygon(polygon).area == 0:
        return None
    return polygon


def page_baseline(points: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    baseline = clip_to_page(points, size)
    return baseline if len(baseline) >= 2 else None
