# SPDX-License-Identifier: MIT
import logging
from collections.abc import Sequence
from configparser import ConfigParser
from dataclasses import dataclass

import numpy as np
import torch
from pero_ocr.core.layout import PageLayout, RegionLayout, TextLine
from pero_ocr.document_ocr import page_parser
from pero_ocr.ocr_engine.pytorch_ocr_engine import PytorchEngineLineOCR

from .compat import checkpoints_on
from .engines import EngineError
from .geometry import baseline_from_polygon, line_heights, polyline_length, without_repeats
from .model import Line

logger = logging.getLogger(__name__)

CPU_BATCH_PIXELS = 3840
CUDA_BATCH_PIXELS = 15360
MAX_CROP_WIDTH = 16384
MIN_CROP_WIDTH = 4
MIN_BATCH_FILL = 0.75
WIDTH_STEP = 32


@dataclass(frozen=True)
class Transcription:
    text: str
    confidence: float


def complete_geometry(line: Line) -> bool:
    if line.baseline is not None:
        line.baseline = without_repeats(line.baseline)
    if line.baseline is None or len(line.baseline) < 2:
        if line.polygon is None:
            return False
        line.baseline = baseline_from_polygon(line.polygon)
    if line.heights is None:
        if line.polygon is None:
            return False
        line.heights = line_heights(line.baseline, line.polygon)
    return polyline_length(line.baseline) > 0 and sum(line.heights) >= 1


def decode_ctc(ids: np.ndarray, probabilities: np.ndarray, characters: Sequence[str]) -> Transcription:
    starts = np.flatnonzero(np.concatenate([[True], ids[1:] != ids[:-1]]))
    blank = len(characters) - 1
    text = ''.join(characters[index] for index in ids[starts] if index != blank)
    return Transcription(text, float(np.maximum.reduceat(probabilities, starts).min()))


def plan_batches(widths: Sequence[int], max_pixels: int, padding: int) -> list[list[int]]:
    batches: list[list[int]] = []
    batch_width = 0
    for index in sorted(range(len(widths)), key=lambda item: -widths[item]):
        width = int(np.ceil(widths[index] / WIDTH_STEP)) * WIDTH_STEP + 2 * padding
        if batches and width >= MIN_BATCH_FILL * batch_width and batch_width * (len(batches[-1]) + 1) <= max_pixels:
            batches[-1].append(index)
        else:
            batches.append([index])
            batch_width = width
    return batches


class CtcRecognizer:
    def __init__(self, engine: PytorchEngineLineOCR, batch_pixels: int) -> None:
        self.engine = engine
        self.batch_pixels = batch_pixels
        engine.model.eval()

    def recognize(self, crops: Sequence[np.ndarray]) -> list[Transcription]:
        result: list[Transcription | None] = [None] * len(crops)
        batches = plan_batches([crop.shape[1] for crop in crops], self.batch_pixels, self.engine.line_padding_px)
        while batches:
            batch = batches.pop(0)
            try:
                transcriptions = self._run([crops[index] for index in batch])
            except torch.cuda.OutOfMemoryError:
                if len(batch) == 1:
                    raise
                torch.cuda.empty_cache()
                logger.warning(f'Out of GPU memory with {len(batch)} lines; retrying with smaller batches')
                middle = len(batch) // 2
                batches[:0] = [batch[:middle], batch[middle:]]
                continue
            for index, transcription in zip(batch, transcriptions, strict=True):
                result[index] = transcription
        return [transcription for transcription in result if transcription is not None]

    def _run(self, crops: Sequence[np.ndarray]) -> list[Transcription]:
        engine = self.engine
        padding = engine.line_padding_px
        width = max(int(np.ceil(crop.shape[1] / WIDTH_STEP)) * WIDTH_STEP for crop in crops)
        batch = np.zeros((len(crops), engine.line_px_height, width + 2 * padding, 3), dtype=np.uint8)
        for data, crop in zip(batch, crops, strict=True):
            data[:, padding : padding + crop.shape[1]] = crop
        tensor = torch.from_numpy(batch).to(engine.device).float().div_(255.0).permute(0, 3, 1, 2)
        arguments = [tensor]
        if engine.embed_id is not None:
            arguments.append(torch.full((len(crops),), engine.embed_id, dtype=torch.long, device=engine.device))
        probabilities, ids = engine.model(*arguments).softmax(dim=1).max(dim=1)
        return [
            decode_ctc(line_ids, line_probabilities, engine.characters)
            for line_ids, line_probabilities in zip(ids.cpu().numpy(), probabilities.cpu().numpy(), strict=True)
        ]


class PeroRecognizer:
    def __init__(self, ocr: page_parser.PageOCR, decoder: page_parser.PageDecoder | None, batch_pixels: int) -> None:
        self.ocr = ocr
        self.decoder = decoder
        self.batch_pixels = batch_pixels

    def recognize(self, crops: Sequence[np.ndarray]) -> list[Transcription]:
        engine = self.ocr.ocr_engine
        widest = max(crop.shape[1] for crop in crops) + WIDTH_STEP + 2 * engine.line_padding_px
        if self.ocr.provides_ctc_logits:
            engine.max_input_horizontal_pixels = max(self.batch_pixels, widest)
        region = RegionLayout('r', np.zeros((4, 2)))
        region.lines = [TextLine(id=str(index), crop=crop) for index, crop in enumerate(crops)]
        layout = PageLayout(id='page')
        layout.regions = [region]
        layout = self.ocr.process_page(None, layout)
        if self.decoder is not None:
            layout = self.decoder.process_page(layout)
        return [
            Transcription(line.transcription or '', float(page_parser.PageParser.compute_line_confidence(line)))
            for line in region.lines
        ]


class LineRecognizer:
    def __init__(
        self, config: ConfigParser, device: torch.device, directory: str, *, batch_pixels: int | None = None
    ) -> None:
        if not config.has_section('LINE_CROPPER') or not config.has_section('OCR'):
            raise EngineError('Recognition needs the [LINE_CROPPER] and [OCR] sections')
        self.cropper = page_parser.line_cropper_factory(config, config_path=directory).crop_engine
        with checkpoints_on(device):
            ocr = page_parser.ocr_factory(config, device, config_path=directory)
        if ocr.ocr_engine.line_px_height != self.cropper.line_height:
            raise EngineError(
                f'[LINE_CROPPER] LINE_HEIGHT is {self.cropper.line_height}, '
                f'but the OCR network expects {ocr.ocr_engine.line_px_height}'
            )
        pixels = batch_pixels or (CUDA_BATCH_PIXELS if device.type == 'cuda' else CPU_BATCH_PIXELS)
        decoder = None
        if config['PAGE_PARSER'].getboolean('RUN_DECODER', fallback=False):
            decoder = page_parser.page_decoder_factory(config, device, config_path=directory)
        if decoder is None and ocr.provides_ctc_logits:
            self.recognizer = CtcRecognizer(ocr.ocr_engine, pixels)
        else:
            self.recognizer = PeroRecognizer(ocr, decoder, pixels)

    def recognize(self, image: np.ndarray, lines: Sequence[Line]) -> list[Line]:
        crops = []
        readable = []
        unreadable = []
        for line in lines:
            crop = self._crop(image, line)
            if crop is None:
                unreadable.append(line)
            else:
                crops.append(crop)
                readable.append(line)
        if crops:
            for line, transcription in zip(readable, self.recognizer.recognize(crops), strict=True):
                line.text = transcription.text.strip()
                line.confidence = transcription.confidence
        return unreadable

    def _crop(self, image: np.ndarray, line: Line) -> np.ndarray | None:
        if not complete_geometry(line):
            return None
        heights = line.heights
        width = polyline_length(line.baseline) * self.cropper.line_height / (sum(heights) * self.cropper.scale)
        if not MIN_CROP_WIDTH <= width <= MAX_CROP_WIDTH:
            return None
        try:
            coordinates = self.cropper.get_crop_inputs(line.baseline, heights, self.cropper.line_height)
            crop = self.cropper.fast_remap(image, coordinates)
        except Exception:
            logger.debug(f'Could not crop line {line.id}', exc_info=True)
            return None
        return crop if crop is not None and crop.size and crop.shape[1] <= MAX_CROP_WIDTH else None
