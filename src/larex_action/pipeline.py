# SPDX-License-Identifier: MIT
import contextlib
import io
import logging
import math
import time
from importlib.metadata import version

import cv2
import numpy as np
import torch

from .compat import apply_patches, check_supported, seed
from .engines import Engine, EngineError, has_layout, has_recognition
from .errors import PageInputError, PageSkipped
from .images import decode_image
from .jobs import PageJob, PageOptions, PageResult, Stage
from .layout import LayoutAnalyzer
from .model import Region
from .page_xml import PageDocument
from .recognition import LineRecognizer

logger = logging.getLogger(__name__)

CREATOR = f'pero-ocr {version("pero-ocr")}'
MAX_ASPECT_DEVIATION = 0.02
MAX_LISTED_IDS = 20


def resolve_device(name: str) -> torch.device:
    available = torch.cuda.is_available()
    if name == 'auto':
        if torch.version.cuda and not available:
            logger.warning('No usable GPU found, running on the CPU')
        return torch.device('cuda' if available else 'cpu')
    if name.startswith('cuda') and not available:
        raise EngineError(f'PERO_DEVICE is {name}, but no CUDA device is available')
    return torch.device(name)


def listed(ids: list[str]) -> str:
    suffix = f' and {len(ids) - MAX_LISTED_IDS} more' if len(ids) > MAX_LISTED_IDS else ''
    return ', '.join(ids[:MAX_LISTED_IDS]) + suffix


@contextlib.contextmanager
def quiet_output():
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            yield
    finally:
        if text := output.getvalue().strip():
            logger.debug(f'pero-ocr output:\n{text}')


class Pipeline:
    def __init__(
        self,
        engine: Engine,
        stages: frozenset[Stage],
        *,
        device: str = 'auto',
        cpu_threads: int | None = None,
        batch_pixels: int | None = None,
        max_megapixels: int = 200,
    ) -> None:
        if cpu_threads:
            torch.set_num_threads(cpu_threads)
            cv2.setNumThreads(cpu_threads)
        torch.jit.set_fusion_strategy([('STATIC', 0), ('DYNAMIC', 0)])
        apply_patches()

        self.max_megapixels = max_megapixels
        self.layout: LayoutAnalyzer | None = None
        self.recognizer: LineRecognizer | None = None
        self.device = resolve_device(device)
        config = engine.read_config()
        check_supported(config)
        directory = str(engine.directory)
        started = time.monotonic()
        try:
            with quiet_output(), torch.inference_mode():
                if 'layout' in stages and has_layout(config):
                    self.layout = LayoutAnalyzer(config, self.device, directory)
                if 'recognition' in stages and has_recognition(config):
                    self.recognizer = LineRecognizer(config, self.device, directory, batch_pixels=batch_pixels)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise EngineError(f'Engine "{engine.name}" cannot be loaded on {self.device}: {exc}') from exc
        logger.info(f'Loaded engine {engine.name} on {self.device} in {time.monotonic() - started:.1f} s')

    def process(self, job: PageJob) -> PageResult:
        seed()
        if job.first_of_run and self.layout is not None:
            self.layout.reset_scale()
        with quiet_output(), torch.inference_mode():
            return self._process(job)

    def _process(self, job: PageJob) -> PageResult:
        options = job.options
        warnings: list[str] = []
        document = PageDocument.parse(job.xml) if job.xml else None
        image = decode_image(job.image, self.max_megapixels)
        height, width = image.shape[:2]
        if document is None:
            document = PageDocument.create(job.image_name, (width, height), CREATOR)
        scale = self._fit(document, (width, height))

        regions = document.regions_in_scope(job.target, region_ids=job.region_ids, line_ids=job.line_ids)
        self._scale(regions, 1 / scale)
        if job.target != 'PAGE' and not regions:
            raise PageInputError('The selected elements are not part of the PAGE XML')

        new_layout = job.target == 'PAGE' and (
            options.segmentation == 'regions' or (options.segmentation == 'auto' and not regions)
        )
        segmented: list[Region] = []
        if new_layout:
            regions = self._require_layout().detect_page(image, warnings)
            if not any(region.lines for region in regions):
                raise PageSkipped('No text found on the page')
        elif job.target != 'TEXT_LINE' and options.segmentation != 'keep':
            segmented = self._segment_lines(image, regions, options, warnings)

        lines = [line for region in regions for line in region.lines]
        if not lines:
            raise PageSkipped('No text lines in scope')
        if not options.recognition and not new_layout and not segmented:
            raise PageSkipped('All text regions already have lines')
        if options.recognition:
            unreadable = self._require_recognizer().recognize(image, lines)
            if unreadable_ids := [line.id for line in unreadable if not line.detected]:
                warnings.append(f'{len(unreadable_ids)} line(s) have unusable geometry: {listed(unreadable_ids)}')
            self._drop_uncertain(regions, options.min_confidence)
            segmented = [region for region in segmented if region.lines]
            if new_layout:
                regions = [region for region in regions if region.lines]
                if not regions:
                    raise PageSkipped('No line reached the minimum confidence')

        self._scale(regions, scale)
        if new_layout:
            document.replace_text_regions(regions)
        for region in segmented:
            document.replace_lines(region)
        written = 0
        for region in regions:
            for line in region.lines:
                if line.element is None:
                    continue
                written += 1
                if options.recognition:
                    document.write_text(line, options.text_index)
            if options.recognition or region in segmented:
                document.refresh_region_text(region, options.text_index)
        document.touch()
        return PageResult(xml=document.serialize(), warnings=warnings, lines=written)

    def _segment_lines(
        self, image: np.ndarray, regions: list[Region], options: PageOptions, warnings: list[str]
    ) -> list[Region]:
        candidates = [region for region in regions if options.segmentation == 'lines' or not region.lines]
        targets = [region for region in candidates if region.polygon is not None]
        if len(targets) < len(candidates):
            warnings.append(f'{len(candidates) - len(targets)} text region(s) have no usable outline')
        if not targets:
            return []
        found = self._require_layout().detect_lines(image, [region.polygon for region in targets])
        segmented = []
        empty = []
        for region, lines in zip(targets, found, strict=True):
            if lines:
                region.lines = lines
                segmented.append(region)
            else:
                empty.append(region.element.get('id') or '?')
        if empty:
            warnings.append(f'No lines found in {len(empty)} text region(s): {listed(empty)}')
        return segmented

    @staticmethod
    def _drop_uncertain(regions: list[Region], threshold: float) -> None:
        if threshold <= 0:
            return
        for region in regions:
            region.lines = [
                line
                for line in region.lines
                if not line.detected or (line.confidence is not None and line.confidence >= threshold)
            ]

    @staticmethod
    def _fit(document: PageDocument, image_size: tuple[int, int]) -> np.ndarray:
        if not all(document.size):
            document.size = image_size
        scale = np.asarray(document.size, dtype=float) / np.asarray(image_size, dtype=float)
        if abs(math.log(scale[0] / scale[1])) > MAX_ASPECT_DEVIATION:
            raise PageInputError(
                f'The image is {image_size[0]}x{image_size[1]} pixels, '
                f'which does not fit the PAGE XML size {document.size[0]}x{document.size[1]}'
            )
        return scale

    @staticmethod
    def _scale(regions: list[Region], factor: np.ndarray) -> None:
        if np.all(factor == 1):
            return
        for region in regions:
            if region.polygon is not None:
                region.polygon = region.polygon * factor
            for line in region.lines:
                line.baseline = line.baseline * factor if line.baseline is not None else None
                line.polygon = line.polygon * factor if line.polygon is not None else None

    def _require_layout(self) -> LayoutAnalyzer:
        if self.layout is None or not self.layout.detects_lines:
            raise PageInputError('The engine has no line detection; choose another engine or segment the page first')
        return self.layout

    def _require_recognizer(self) -> LineRecognizer:
        if self.recognizer is None:
            raise PageInputError('The engine has no text recognition; choose another engine')
        return self.recognizer
