# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import torch
from pero_ocr.core.layout import RegionLayout, TextLine
from shapely import LineString
from stub_engine import draw_page, line_width

from larex_action.compat import apply_patches, seed
from larex_action.engines import Engine
from larex_action.layout import LayoutAnalyzer, _drop_fragments


def box(left, top, right, bottom):
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)


def text_line(name, left, right, y=50.0):
    return TextLine(
        id=name,
        baseline=np.array([[left, y], [right, y]]),
        polygon=box(left, y - 15, right, y + 5),
        heights=[15, 5],
    )


def region(name, outline, *lines):
    result = RegionLayout(name, outline)
    result.lines = list(lines)
    return result


def names(*regions):
    return [[line.id for line in item.lines] for item in regions]


@pytest.fixture(scope='module')
def analyzer(model_dir):
    apply_patches()
    engine = Engine('full', model_dir / 'full' / 'config.ini')
    return LayoutAnalyzer(engine.read_config(), torch.device('cpu'), str(engine.directory))


def test_lines_are_assigned_to_their_regions(analyzer):
    seed()
    image = draw_page([('ab', 20, 60), ('bba', 20, 100), ('aab', 330, 100)])
    outlines = [box(320, 70, 520, 120), box(10, 30, 250, 120), box(10, 300, 250, 350)]
    with torch.inference_mode():
        found = analyzer.detect_lines(image, outlines)
    assert [len(lines) for lines in found] == [1, 2, 0]
    baseline = found[0][0].baseline
    assert baseline[0, 0] == pytest.approx(330, abs=6)
    assert baseline[-1, 0] == pytest.approx(330 + line_width('aab'), abs=6)
    assert baseline[:, 1].mean() == pytest.approx(100, abs=2)


def test_pages_without_text_have_no_regions(analyzer):
    with torch.inference_mode():
        assert analyzer.detect_page(draw_page([]), []) == []


def test_scale_is_reset_after_an_empty_page(analyzer, monkeypatch):
    network = analyzer.parsers[0].engine.parsenet
    monkeypatch.setattr(network, 'last_downsample', 8)
    with torch.inference_mode():
        regions = analyzer.detect_page(draw_page([('ab', 20, 60)]), [])
    assert regions
    assert network.last_downsample == network.init_downsample


def test_fragments_clipped_by_a_region_border_are_dropped():
    item = region('r0', box(90, 0, 300, 100), text_line('r0-1', 90, 110), text_line('r0-2', 100, 280, 80.0))
    detections = [LineString([(0, 50), (110, 50)]), LineString([(100, 80), (280, 80)])]
    _drop_fragments([item], detections)
    assert names(item) == [['r0-2']]


def test_short_lines_are_kept():
    item = region('r0', box(0, 0, 300, 100), text_line('r0-1', 100, 115))
    _drop_fragments([item], [LineString([(100, 50), (115, 50)])])
    assert names(item) == [['r0-1']]


def test_lines_shared_by_overlapping_regions_stay_in_the_smaller_one():
    outer = region('r0', box(0, 0, 500, 500), text_line('r0-1', 20, 200))
    inner = region('r1', box(10, 10, 220, 100), text_line('r1-1', 20, 200))
    _drop_fragments([outer, inner], [LineString([(20, 50), (200, 50)])])
    assert names(outer, inner) == [[], ['r1-1']]


def test_lines_split_between_neighbours_are_kept_in_both():
    left = region('r0', box(0, 0, 100, 100), text_line('r0-1', 20, 100))
    right = region('r1', box(100, 0, 220, 100), text_line('r1-1', 100, 200))
    _drop_fragments([left, right], [LineString([(20, 50), (200, 50)])])
    assert names(left, right) == [['r0-1'], ['r1-1']]
