# SPDX-License-Identifier: MIT
import importlib.util
import multiprocessing
import sys
from configparser import ConfigParser

import numpy as np
import pytest
import shapely
import torch
from pero_ocr.core import crop_engine
from pero_ocr.document_ocr import page_parser
from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine

from larex_action.compat import apply_patches, check_supported
from larex_action.engines import Engine, EngineError


@pytest.fixture(scope='module')
def upstream_cropper():
    spec = importlib.util.spec_from_file_location('upstream_crop_engine', crop_engine.__file__)
    module = importlib.util.module_from_spec(spec)
    modules = pytest.MonkeyPatch()
    modules.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    yield module.EngineLineCropper(line_height=40, poly=2, scale=1)
    modules.undo()


@pytest.mark.parametrize('number', range(10))
def test_crops_match_pero(upstream_cropper, number):
    apply_patches()
    patched = crop_engine.EngineLineCropper(line_height=40, poly=2, scale=1)
    rng = np.random.default_rng(number)
    image = rng.integers(0, 255, (300, 900, 3), dtype=np.uint8)
    count = int(rng.integers(2, 8))
    baseline = np.stack(
        [np.sort(rng.choice(np.arange(20, 880), count, replace=False)), rng.integers(120, 160, count)], axis=1
    ).astype(float)
    heights = [float(rng.integers(10, 50)), float(rng.integers(3, 20))]
    expected = upstream_cropper.fast_remap(image, upstream_cropper.get_crop_inputs(baseline, heights, 40))
    actual = patched.fast_remap(image, patched.get_crop_inputs(baseline, heights, 40))
    assert actual.shape == expected.shape
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 1


def test_split_region_outlines_become_separate_polygons(monkeypatch):
    def filter_polygons(_self, _polygons, _textlines):
        left, right = shapely.box(0, 0, 10, 10), shapely.box(20, 0, 30, 10)
        return [
            shapely.MultiPolygon([left, right]),
            shapely.GeometryCollection([left, shapely.LineString([(0, 0), (5, 5)])]),
            shapely.Polygon(),
        ]

    monkeypatch.setattr(LayoutEngine, 'filter_polygons', filter_polygons)
    apply_patches()
    result = LayoutEngine.filter_polygons(None, [], [])
    assert [geometry.geom_type for geometry in result] == ['Polygon'] * 3


def test_patches_are_applied_once():
    apply_patches()
    patched = LayoutEngine.filter_polygons
    apply_patches()
    assert LayoutEngine.filter_polygons is patched


def test_no_process_is_started_for_a_parser(model_dir):
    apply_patches()
    engine = Engine('full', model_dir / 'full' / 'config.ini')
    page_parser.PageOCR(engine.read_config()['OCR'], device=torch.device('cpu'), config_path=str(engine.directory))
    assert multiprocessing.active_children() == []


def test_unsupported_line_stretching_is_rejected():
    config = ConfigParser()
    config.read_string('[LAYOUT_PARSER_2]\nMETHOD = LINE_POSTPROCESSING\nSTRETCH_LINES = max\n')
    with pytest.raises(EngineError, match='STRETCH_LINES'):
        check_supported(config)
    config.set('LAYOUT_PARSER_2', 'STRETCH_LINES', '10')
    check_supported(config)
