# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field

import numpy as np
from lxml import etree


@dataclass
class Line:
    baseline: np.ndarray | None = None
    polygon: np.ndarray | None = None
    heights: tuple[float, float] | None = None
    text: str | None = None
    confidence: float | None = None
    element: etree._Element | None = None

    @property
    def detected(self) -> bool:
        return self.element is None

    @property
    def id(self) -> str:
        return (self.element.get('id') if self.element is not None else None) or '?'


@dataclass
class Region:
    polygon: np.ndarray | None = None
    lines: list[Line] = field(default_factory=list)
    element: etree._Element | None = None
