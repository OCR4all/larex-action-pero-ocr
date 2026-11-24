# SPDX-License-Identifier: MIT
import copy
import logging
from collections.abc import Sequence
from configparser import ConfigParser

import numpy as np
import shapely
import torch
from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import page_parser
from pero_ocr.layout_engines.cnn_layout_engine import LayoutEngine
from pero_ocr.layout_engines.naive_sorter import NaiveRegionSorter
from pero_ocr.layout_engines.smart_sorter import SmartRegionSorter

from .engines import MAX_LAYOUT_PARSERS
from .geometry import polyline_length
from .model import Line, Region

logger = logging.getLogger(__name__)

SORTERS = (NaiveRegionSorter, SmartRegionSorter)
LINE_PARSERS = (
    page_parser.LayoutExtractor,
    page_parser.TextlineExtractorSimple,
    page_parser.LineFilter,
    page_parser.LinePostprocessor,
)
SHARED_LINE_OVERLAP = 0.5
FRAGMENT_HEIGHTS = 1.5
FRAGMENT_SHARE = 0.5
SOURCE_TOLERANCE = 1.5


def _line(text_line: TextLine) -> Line:
    return Line(
        baseline=np.asarray(text_line.baseline, dtype=float),
        polygon=np.asarray(text_line.polygon, dtype=float),
        heights=(float(text_line.heights[0]), float(text_line.heights[1])),
    )


def _line_height(line: TextLine) -> float:
    return float(line.heights[0] + line.heights[1])


def _recording(engine: LayoutEngine, detections: list[shapely.Geometry]) -> LayoutEngine:
    recorder = copy.copy(engine)

    def detect(image: np.ndarray, rot: int = 0):
        result = engine.detect(image, rot=rot)
        detections.extend(shapely.LineString(baseline) for baseline in result[1])
        return result

    recorder.detect = detect
    return recorder


def _is_clipped_fragment(line: TextLine, detections: Sequence[shapely.Geometry]) -> bool:
    length = polyline_length(line.baseline)
    if length >= FRAGMENT_HEIGHTS * _line_height(line):
        return False
    middle = shapely.line_interpolate_point(shapely.LineString(line.baseline), 0.5, normalized=True)
    sources = [item.length for item in detections if item.distance(middle) < SOURCE_TOLERANCE]
    return bool(sources) and length < FRAGMENT_SHARE * min(sources)


def _drop_fragments(regions: Sequence[RegionLayout], detections: Sequence[shapely.Geometry]) -> None:
    for region in regions:
        region.lines = [line for line in region.lines if not _is_clipped_fragment(line, detections)]

    by_detection: dict[str, list[tuple[RegionLayout, TextLine]]] = {}
    for region in regions:
        for line in region.lines:
            by_detection.setdefault(line.id.rsplit('-', 1)[-1], []).append((region, line))
    for pieces in by_detection.values():
        pieces.sort(key=lambda piece: shapely.Polygon(piece[0].polygon).area)
        kept: list[shapely.Geometry] = []
        for region, line in pieces:
            shape = shapely.LineString(line.baseline)
            duplicate = any(
                shapely.intersection(shape, other.buffer(1)).length
                > SHARED_LINE_OVERLAP * min(shape.length, other.length)
                for other in kept
            )
            if duplicate:
                region.lines = [item for item in region.lines if item is not line]
            else:
                kept.append(shape)


class LayoutAnalyzer:
    def __init__(self, config: ConfigParser, device: torch.device, directory: str) -> None:
        self.parsers = [
            page_parser.layout_parser_factory(config, device, config_path=directory, order=order)
            for order in range(1, MAX_LAYOUT_PARSERS + 1)
            if config.has_section(f'LAYOUT_PARSER_{order}')
        ]

    @property
    def _networks(self):
        return [parser.engine.parsenet for parser in self.parsers if isinstance(parser, page_parser.LayoutExtractor)]

    @property
    def detects_lines(self) -> bool:
        return any(isinstance(parser, LINE_PARSERS[:2]) for parser in self.parsers)

    def reset_scale(self) -> bool:
        changed = False
        for network in self._networks:
            changed = changed or network.last_downsample != network.init_downsample
            network.last_downsample = network.init_downsample
        return changed

    def detect_page(self, image: np.ndarray, warnings: list[str]) -> list[Region]:
        layout = self._detect_page(image, warnings)
        if not layout.regions and self.reset_scale():
            layout = self._detect_page(image, warnings)
        return [
            Region(polygon=np.asarray(region.polygon, dtype=float), lines=[_line(line) for line in region.lines])
            for region in layout.regions
        ]

    def detect_lines(self, image: np.ndarray, polygons: Sequence[np.ndarray]) -> list[list[Line]]:
        lines = self._detect_lines(image, polygons)
        if not any(lines) and self.reset_scale():
            lines = self._detect_lines(image, polygons)
        return lines

    def _detect_page(self, image: np.ndarray, warnings: list[str]) -> PageLayout:
        layout = PageLayout(id='page', page_size=image.shape[:2])
        for parser in self.parsers:
            if isinstance(parser, SORTERS):
                self._sort(parser, image, layout, warnings)
            else:
                layout = parser.process_page(image, layout)
        return layout

    def _detect_lines(self, image: np.ndarray, polygons: Sequence[np.ndarray]) -> list[list[Line]]:
        detections: list[shapely.Geometry] = []
        regions = [RegionLayout(f'r{index}', np.asarray(polygon)) for index, polygon in enumerate(polygons)]
        layout = PageLayout(id='page', page_size=image.shape[:2])
        layout.regions = list(regions)
        for parser in self.parsers:
            if not isinstance(parser, LINE_PARSERS):
                continue
            if isinstance(parser, page_parser.LayoutExtractor):
                parser = copy.copy(parser)
                parser.detect_regions = False
                parser.engine = _recording(parser.engine, detections)
            layout = parser.process_page(image, layout)
        remaining = {region.id: region for region in layout.regions}
        _drop_fragments(list(remaining.values()), detections)
        return [
            [_line(line) for line in remaining[region.id].lines] if region.id in remaining else [] for region in regions
        ]

    @staticmethod
    def _sort(
        parser: NaiveRegionSorter | SmartRegionSorter, image: np.ndarray, layout: PageLayout, warnings: list[str]
    ) -> None:
        try:
            ordered = parser.process_page(image, copy.deepcopy(layout))
            by_id = {region.id: region for region in layout.regions}
            order = [region.id for region in ordered.regions]
            if sorted(order) != sorted(by_id):
                raise ValueError('Region sorter lost or repeated regions')
        except Exception:
            logger.exception('Region sorting failed')
            warnings.append('The reading order could not be determined; regions are in detection order')
            return
        layout.regions = [by_id[region_id] for region_id in order]
