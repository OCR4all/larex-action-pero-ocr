# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from larex_action.geometry import (
    baseline_from_polygon,
    line_heights,
    page_baseline,
    page_polygon,
    polyline_length,
    sample_polyline,
    without_repeats,
)


def box(left, top, right, bottom):
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)


def rotate(points, degrees):
    angle = np.deg2rad(degrees)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return points @ matrix.T


def test_baseline_of_a_box():
    baseline = baseline_from_polygon(box(10, 20, 110, 40))
    np.testing.assert_allclose(baseline, [[10, 36], [110, 36]], atol=1e-6)


@pytest.mark.parametrize('degrees', [-30, -5, 5, 30])
def test_baseline_of_a_rotated_box(degrees):
    baseline = baseline_from_polygon(rotate(box(0, 0, 100, 20), degrees))
    np.testing.assert_allclose(baseline, rotate(np.array([[0, 16], [100, 16]], dtype=float), degrees), atol=1e-6)


def test_baseline_of_vertical_text_runs_down():
    baseline = baseline_from_polygon(box(0, 0, 20, 200))
    assert baseline[0, 1] < baseline[1, 1]


def test_baseline_of_a_single_character_stays_horizontal():
    baseline = baseline_from_polygon(box(0, 0, 10, 18))
    assert baseline[1, 0] - baseline[0, 0] == pytest.approx(10)


def test_heights_from_polygon_and_baseline():
    ascender, descender = line_heights(np.array([[0.0, 15.0], [100.0, 15.0]]), box(0, 0, 100, 20))
    assert (ascender, descender) == (pytest.approx(15), pytest.approx(5))


def test_heights_ignore_a_single_peak():
    polygon = np.array([[0, 5], [48, 5], [50, -60], [52, 5], [100, 5], [100, 20], [0, 20]], dtype=float)
    ascender, _ = line_heights(np.array([[0.0, 15.0], [100.0, 15.0]]), polygon)
    assert ascender < 15


def test_heights_fall_back_when_the_baseline_is_outside():
    ascender, descender = line_heights(np.array([[500.0, 500.0], [600.0, 500.0]]), box(0, 0, 100, 20))
    assert (ascender, descender) == (pytest.approx(16), pytest.approx(4))


def test_heights_of_a_self_intersecting_polygon():
    polygon = np.array([[0, 0], [100, 20], [100, 0], [0, 20]], dtype=float)
    ascender, descender = line_heights(np.array([[0.0, 12.0], [100.0, 12.0]]), polygon)
    assert ascender > 0
    assert descender >= 0


def test_polyline_helpers():
    line = np.array([[0, 0], [0, 0], [3, 4], [3, 4], [6, 8]], dtype=float)
    assert polyline_length(line) == 10
    assert len(without_repeats(line)) == 3
    np.testing.assert_allclose(sample_polyline(without_repeats(line), 2), [[1.5, 2], [4.5, 6]])


def test_page_geometry_is_clipped_and_rounded():
    polygon = page_polygon(box(-5.4, 2.6, 250, 20.2), (200, 100))
    np.testing.assert_array_equal(polygon, [[0, 3], [199, 3], [199, 20], [0, 20]])
    assert page_polygon(box(300, 0, 400, 10), (200, 100)) is None
    assert page_baseline(np.array([[300.0, 5.0], [400.0, 5.0]]), (200, 100)) is None


def test_closed_polygons_are_opened():
    closed = np.vstack([box(0, 0, 10, 10), [[0, 0]]])
    assert len(page_polygon(closed, (200, 100))) == 4
