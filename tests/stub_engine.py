# SPDX-License-Identifier: MIT
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ASCENDER = 14
DESCENDER = 4
BLOCK = 24
GAP = 12
COLOURS = {'a': (0, 0, 255), 'b': (0, 255, 0)}


class StubParseNet(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blue = ((x[:, 0] - x[:, 1:].mean(dim=1)) > 0.5).float()
        ones = torch.ones_like(blue)
        maps = torch.stack([ones * 14.0, ones * 4.0, blue, ones * 0.0, ones * 0.0], dim=1)
        return maps, maps


class StubOcr(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        columns = x.mean(dim=2)
        columns = columns[:, :, : columns.shape[2] // 4 * 4].reshape(x.shape[0], 3, -1, 4).mean(dim=3)
        red = columns[:, 2] - columns[:, :2].mean(dim=1)
        green = columns[:, 1] - 0.5 * (columns[:, 0] + columns[:, 2])
        blank = torch.full_like(red, 3.0)
        space = torch.full_like(red, -3.0)
        return torch.stack([red * 20.0, green * 20.0, space, blank], dim=1)


def write_engine(directory: Path, *, layout: bool = True, recognition: bool = True) -> Path:
    directory.mkdir(parents=True)
    config = [
        '[PAGE_PARSER]',
        f'RUN_LAYOUT_PARSER = {"yes" if layout else "no"}',
        'RUN_LINE_CROPPER = yes',
        f'RUN_OCR = {"yes" if recognition else "no"}',
        'RUN_DECODER = no',
    ]
    if layout:
        torch.jit.script(StubParseNet()).save(directory / 'parsenet.pt.cpu')
        config += [
            '[LAYOUT_PARSER_1]',
            'METHOD = LAYOUT_CNN',
            'DETECT_LINES = yes',
            'DETECT_REGIONS = yes',
            'MERGE_LINES = no',
            'ADJUST_HEIGHTS = no',
            'MODEL_PATH = ./parsenet.pt',
            'MAX_MEGAPIXELS = 5',
            'USE_CPU = yes',
            'DOWNSAMPLE = 1',
            'ADAPTIVE_DOWNSAMPLE = no',
            'DETECTION_THRESHOLD = 0.2',
            'LINE_END_WEIGHT = 1.0',
            'VERTICAL_LINE_CONNECTION_RANGE = 3',
            'SMOOTH_LINE_PREDICTIONS = no',
            '[LAYOUT_PARSER_2]',
            'METHOD = REGION_SORTER_SMART',
        ]
    if recognition:
        torch.jit.script(StubOcr()).save(directory / 'ocr.pt.cpu')
        engine = {
            'exported': 'true',
            'checkpoint': './ocr.pt',
            'line_px_height': 40,
            'line_vertical_scale': 1.25,
            'net_name': 'STUB',
            'characters': ['a', 'b', ' '],
        }
        (directory / 'ocr_engine.json').write_text(json.dumps(engine))
        config += [
            '[LINE_CROPPER]',
            'INTERP = 2',
            'LINE_SCALE = 1.0',
            'LINE_HEIGHT = 40',
            '[OCR]',
            'OCR_JSON = ./ocr_engine.json',
            'USE_CPU = yes',
        ]
    (directory / 'config.ini').write_text('\n'.join(config) + '\n')
    return directory


def line_width(text: str) -> int:
    return len(text) * (BLOCK + GAP) + GAP


def draw_page(lines: list[tuple[str, int, int]], size: tuple[int, int] = (600, 400)) -> np.ndarray:
    image = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    for text, left, baseline in lines:
        for position, character in enumerate(text):
            start = left + GAP + position * (BLOCK + GAP)
            image[baseline - ASCENDER + 2 : baseline - 1, start : start + BLOCK] = COLOURS[character]
        image[baseline, left : left + line_width(text)] = (255, 0, 0)
    return image


def encode(image: np.ndarray, extension: str = '.png') -> bytes:
    return cv2.imencode(extension, image)[1].tobytes()
