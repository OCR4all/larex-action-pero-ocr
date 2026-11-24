# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pero_ocr.ocr_engine.pytorch_ocr_engine import greedy_decode_ctc
from stub_engine import draw_page

from larex_action.engines import find_engine
from larex_action.model import Line
from larex_action.recognition import CtcRecognizer, LineRecognizer, complete_geometry, decode_ctc, plan_batches

CHARACTERS = list('abcdefg') + ['​']


def line(left, top, right, bottom, baseline=True):
    return Line(
        baseline=np.array([[left, bottom - 4], [right, bottom - 4]], dtype=float) if baseline else None,
        polygon=np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float),
    )


@pytest.mark.parametrize('number', range(20))
def test_decoding_matches_pero(number):
    logits = torch.from_numpy(np.random.default_rng(number).normal(0, 3, (1, len(CHARACTERS), 60)).astype(np.float32))
    probabilities, ids = logits.softmax(dim=1).max(dim=1)
    expected = greedy_decode_ctc(logits.clone(), CHARACTERS)[0]
    assert decode_ctc(ids[0].numpy(), probabilities[0].numpy(), CHARACTERS).text == expected


def test_confidence_is_the_weakest_symbol():
    ids = np.array([7, 0, 0, 7, 1])
    result = decode_ctc(ids, np.array([0.99, 0.4, 0.9, 0.95, 0.8]), CHARACTERS)
    assert result.text == 'ab'
    assert result.confidence == pytest.approx(0.8)


@pytest.mark.parametrize('limit', [500, 2000, 8000])
def test_batches_hold_every_line_once_within_the_limit(limit):
    widths = np.random.default_rng(1).integers(20, 1500, 80).tolist()
    batches = plan_batches(widths, limit, 32)
    assert sorted(index for batch in batches for index in batch) == list(range(80))
    for batch in batches:
        widest = max(int(np.ceil(widths[index] / 32)) * 32 + 64 for index in batch)
        assert len(batch) == 1 or widest * len(batch) <= limit


def test_no_lines_need_no_batches():
    assert plan_batches([], 1000, 32) == []


def test_geometry_is_completed_from_the_outline():
    item = line(0, 0, 100, 20, baseline=False)
    assert complete_geometry(item)
    assert item.baseline is not None
    assert sum(item.heights) == pytest.approx(20)


def test_lines_without_outline_are_unusable():
    assert not complete_geometry(Line(baseline=np.array([[0.0, 0.0], [10.0, 0.0]])))
    assert not complete_geometry(Line())


@pytest.fixture(scope='module')
def recognizer(model_dir):
    from larex_action.compat import apply_patches
    from larex_action.engines import Engine

    apply_patches()
    engine = Engine('full', model_dir / 'full' / 'config.ini')
    return LineRecognizer(engine.read_config(), torch.device('cpu'), str(engine.directory))


def test_recognizes_lines_in_their_given_order(recognizer, installed_engines):
    assert find_engine('full').directory == installed_engines / 'full'
    image = draw_page([('ab', 20, 60), ('bbbbaaa', 20, 120), ('a', 20, 180)])
    lines = [line(20, 46, 106, 64), line(20, 106, 290, 124), line(20, 166, 72, 184)]
    assert recognizer.recognize(image, lines) == []
    assert [item.text for item in lines] == ['ab', 'bbbbaaa', 'a']
    assert all(0 < item.confidence <= 1 for item in lines)


def test_lines_without_readable_crop_are_returned(recognizer):
    image = draw_page([('ab', 20, 60)])
    lines = [line(20, 46, 106, 64), line(0, 0, 30000, 20), Line()]
    assert recognizer.recognize(image, lines) == lines[1:]
    assert [item.text for item in lines] == ['ab', None, None]


def test_out_of_memory_splits_the_batch():
    sizes = []

    class Model:
        def eval(self):
            return self

        def __call__(self, batch):
            sizes.append(len(batch))
            if len(batch) > 2:
                raise torch.cuda.OutOfMemoryError
            scores = torch.zeros(len(batch), 3, batch.shape[3] // 4)
            scores[:, 0] = 5
            return scores

    engine = SimpleNamespace(
        model=Model(),
        line_padding_px=32,
        line_px_height=40,
        device=torch.device('cpu'),
        embed_id=None,
        characters=['a', 'b', '​'],
    )
    crops = [np.zeros((40, 100, 3), dtype=np.uint8) for _ in range(5)]
    result = CtcRecognizer(engine, 100_000).recognize(crops)
    assert [item.text for item in result] == ['a'] * 5
    assert sizes == [5, 2, 3, 1, 2]
