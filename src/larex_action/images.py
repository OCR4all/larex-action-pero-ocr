# SPDX-License-Identifier: MIT
import io

import cv2
import numpy as np
from PIL import Image

from .errors import PageInputError


def _check_size(width: int, height: int, max_megapixels: int) -> None:
    if width * height > max_megapixels * 1_000_000:
        raise PageInputError(
            f'Image has {width * height / 1e6:.0f} megapixels, the processor accepts up to {max_megapixels}'
        )


def decode_image(content: bytes, max_megapixels: int) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(io.BytesIO(content)) as source:
            _check_size(*source.size, max_megapixels)
            if 'A' not in source.getbands() and 'transparency' not in source.info:
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if image is not None:
                    return image
            rgba = source.convert('RGBA')
    except PageInputError:
        raise
    except (OSError, SyntaxError, ValueError, cv2.error) as exc:
        raise PageInputError(f'Unreadable page image: {exc}') from exc
    flattened = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
    return cv2.cvtColor(np.asarray(flattened), cv2.COLOR_RGB2BGR)
