# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from typing import Literal

Segmentation = Literal['auto', 'regions', 'lines', 'keep']
Stage = Literal['layout', 'recognition']
Target = Literal['PAGE', 'REGION', 'TEXT_LINE']


@dataclass(frozen=True)
class PageOptions:
    segmentation: Segmentation = 'auto'
    recognition: bool = True
    text_index: int = 1
    min_confidence: float = 0.0


@dataclass
class PageJob:
    image: bytes
    xml: bytes | None
    image_name: str
    options: PageOptions
    target: Target = 'PAGE'
    region_ids: tuple[str, ...] = ()
    line_ids: tuple[str, ...] = ()
    first_of_run: bool = True


@dataclass
class PageResult:
    xml: bytes
    warnings: list[str] = field(default_factory=list)
    lines: int = 0
