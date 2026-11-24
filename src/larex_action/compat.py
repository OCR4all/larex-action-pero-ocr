# SPDX-License-Identifier: MIT
import functools
import random
from collections.abc import Iterator
from configparser import ConfigParser
from contextlib import contextmanager

import cv2
import numpy as np
import shapely
import torch
from pero_ocr.core import crop_engine
from pero_ocr.document_ocr import page_parser
from pero_ocr.layout_engines import cnn_layout_engine, torch_parsenet
from pero_ocr.ocr_engine import transformer_ocr_engine

from .engines import MAX_LAYOUT_PARSERS, EngineError


def _no_pool(*_args: object) -> None:
    return None


def _sample_positions(
    _self: object, forward_mapping: np.ndarray, sample_positions: np.ndarray, sampled_values: np.ndarray
) -> np.ndarray:
    share = sample_positions / forward_mapping[-1]
    return (1 - share) * sampled_values[0] + share * sampled_values[-1]


def _get_maps(self: torch_parsenet.TorchParseNet, img: np.ndarray, downsample: float) -> np.ndarray:
    img = cv2.resize(img, (0, 0), fx=1 / downsample, fy=1 / downsample, interpolation=cv2.INTER_AREA)
    height, width = img.shape[:2]
    canvas = np.zeros((1, int(np.ceil(height / 64)) * 64, int(np.ceil(width / 64)) * 64, 3), dtype=np.uint8)
    canvas[0, :height, :width] = img
    tensor = torch.from_numpy(canvas).to(self.device).float().permute(0, 3, 1, 2) * (1 / 255.0)
    out_map, _ = self.net(tensor)
    return out_map[0].permute(1, 2, 0)[:height, :width].cpu().numpy()


def _single_polygons(filter_polygons):
    @functools.wraps(filter_polygons)
    def wrapper(self, polygons, region_textlines):
        return [
            part
            for geometry in filter_polygons(self, polygons, region_textlines)
            for part in shapely.get_parts(geometry)
            if part.geom_type == 'Polygon' and not part.is_empty
        ]

    return wrapper


def apply_patches() -> None:
    page_parser.Pool = _no_pool
    crop_engine.EngineLineCropper.reverse_line_mapping = _sample_positions
    torch_parsenet.TorchParseNet.get_maps = _get_maps
    if not hasattr(cnn_layout_engine.LayoutEngine.filter_polygons, '__wrapped__'):
        cnn_layout_engine.LayoutEngine.filter_polygons = _single_polygons(
            cnn_layout_engine.LayoutEngine.filter_polygons
        )


def check_supported(config: ConfigParser) -> None:
    for order in range(1, MAX_LAYOUT_PARSERS + 1):
        section = f'LAYOUT_PARSER_{order}'
        if config.get(section, 'STRETCH_LINES', fallback='') == 'max':
            raise EngineError(f'[{section}] STRETCH_LINES = max does not work in pero-ocr 0.7.0')


@contextmanager
def checkpoints_on(device: torch.device) -> Iterator[None]:
    original = transformer_ocr_engine.torch.load
    transformer_ocr_engine.torch.load = functools.partial(original, map_location=device)
    try:
        yield
    finally:
        transformer_ocr_engine.torch.load = original


def seed() -> None:
    random.seed(0)
    np.random.seed(0)
