# SPDX-License-Identifier: MIT
import io

import numpy as np
import pytest
from PIL import Image
from stub_engine import draw_page, encode

from larex_action.errors import PageInputError
from larex_action.images import decode_image


def save(image, file_format, **options):
    buffer = io.BytesIO()
    image.save(buffer, format=file_format, **options)
    return buffer.getvalue()


@pytest.mark.parametrize('extension', ['.png', '.tif', '.bmp', '.webp', '.jp2'])
def test_formats(extension):
    page = draw_page([('ab', 20, 50)])
    decoded = decode_image(encode(page, extension), 10)
    assert decoded.shape == page.shape
    assert np.abs(decoded.astype(int) - page).mean() < 3


@pytest.mark.parametrize('mode', ['1', 'L', 'P', 'CMYK', 'I;16'])
def test_colour_modes(mode):
    content = save(Image.new(mode, (30, 20)), 'TIFF')
    decoded = decode_image(content, 10)
    assert decoded.shape == (20, 30, 3)
    assert decoded.dtype == np.uint8


def test_exif_orientation_is_ignored():
    exif = Image.Exif()
    exif[274] = 6
    content = save(Image.new('RGB', (30, 20), 'white'), 'JPEG', exif=exif)
    assert decode_image(content, 10).shape == (20, 30, 3)


def test_transparent_pixels_become_white():
    image = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (255, 0, 0, 255))
    decoded = decode_image(save(image, 'PNG'), 10)
    assert decoded[0, 0].tolist() == [255, 255, 255]
    assert decoded[1, 1].tolist() == [0, 0, 255]


def test_too_large_images_are_rejected():
    with pytest.raises(PageInputError, match='2 megapixels, the processor accepts up to 1'):
        decode_image(save(Image.new('1', (2000, 1000)), 'PNG'), 1)


@pytest.mark.parametrize('content', [b'', b'not an image', save(Image.new('RGB', (64, 64), 'red'), 'PNG')[:60]])
def test_unreadable_images_are_rejected(content):
    with pytest.raises(PageInputError, match='Unreadable page image'):
        decode_image(content, 10)
