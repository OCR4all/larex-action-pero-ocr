# SPDX-License-Identifier: MIT
import multiprocessing

import numpy as np
import pytest
import torch
from lxml import etree
from page_fixtures import assert_valid, page_xml, text_line, text_region, texts
from stub_engine import draw_page, encode, line_width

from larex_action.engines import Engine, EngineError
from larex_action.errors import PageInputError, PageSkipped
from larex_action.jobs import PageJob, PageOptions
from larex_action.pipeline import Pipeline, resolve_device

LINES = [('ab', 20, 60), ('bba', 20, 100), ('aab', 330, 300)]
IMAGE = draw_page(LINES)


def box(index, left=0, right=None):
    text, start, baseline = LINES[index]
    return start + left, baseline - 14, start + (right if right is not None else line_width(text)), baseline + 4


EMPTY_REGIONS = page_xml(text_region('r1', (10, 30, 250, 120)) + text_region('r2', (320, 270, 500, 320)))
ONE_LINE = page_xml(
    text_region('r1', (10, 30, 250, 120), text_line('l1', box(0))) + text_region('r2', (320, 270, 500, 320))
)


@pytest.fixture(scope='module')
def pipelines(model_dir):
    return {
        name: Pipeline(
            Engine(name, model_dir / name / 'config.ini'), frozenset({'layout', 'recognition'}), device='cpu'
        )
        for name in ('full', 'layout-only', 'recognition-only')
    }


def run(pipeline, xml=None, image=IMAGE, target='PAGE', regions=(), lines=(), **options):
    content = image if isinstance(image, bytes) else encode(image)
    result = pipeline.process(
        PageJob(content, xml, 'page.png', PageOptions(**options), target, tuple(regions), tuple(lines))
    )
    assert_valid(result.xml)
    return result


def found(result, index=1):
    return sorted(text for text in texts(result.xml, index).values() if text)


def test_new_page_is_segmented_and_recognized(pipelines):
    result = run(pipelines['full'])
    assert found(result) == ['aab', 'ab', 'bba']
    assert result.lines == 3
    root = etree.fromstring(result.xml)
    page = root.find('{*}Page')
    assert (page.get('imageWidth'), page.get('imageHeight'), page.get('imageFilename')) == ('600', '400', 'page.png')
    regions = [region.get('id') for region in root.iter('{*}TextRegion')]
    assert [item.get('regionRef') for item in root.iter('{*}RegionRefIndexed')] == regions


def test_results_are_reproducible(pipelines):
    first, second = run(pipelines['full']), run(pipelines['full'])
    strip = lambda content: [element.attrib for element in etree.fromstring(content).iter('{*}Coords')]  # noqa: E731
    assert strip(first.xml) == strip(second.xml)


def test_segmentation_only(pipelines):
    result = run(pipelines['layout-only'], recognition=False)
    assert result.lines == 3
    assert etree.fromstring(result.xml).find('.//{*}TextEquiv') is None


def test_auto_fills_only_regions_without_lines(pipelines):
    result = run(pipelines['full'], ONE_LINE)
    assert found(result) == ['aab', 'ab']
    assert list(texts(result.xml)) == ['l1', 'r2-l1']


def test_lines_replaces_lines_but_not_regions(pipelines):
    result = run(pipelines['full'], ONE_LINE, segmentation='lines')
    assert found(result) == ['aab', 'ab', 'bba']
    assert [region.get('id') for region in etree.fromstring(result.xml).iter('{*}TextRegion')] == ['r1', 'r2']
    assert 'l1' not in texts(result.xml)


def test_regions_replaces_everything(pipelines):
    result = run(pipelines['full'], ONE_LINE, segmentation='regions')
    assert found(result) == ['aab', 'ab', 'bba']
    assert 'l1' not in texts(result.xml)


def test_keep_recognizes_existing_lines(pipelines):
    result = run(pipelines['recognition-only'], ONE_LINE, segmentation='keep', text_index=3)
    assert texts(result.xml, 3) == {'l1': 'ab'}


def test_lines_without_baseline_are_recognized(pipelines):
    xml = page_xml(text_region('r1', (10, 30, 250, 120), text_line('l1', box(0), baseline=False)))
    assert texts(run(pipelines['full'], xml).xml) == {'l1': 'ab'}


def test_selected_regions(pipelines):
    result = run(pipelines['full'], EMPTY_REGIONS, target='REGION', regions=['r2'])
    assert list(texts(result.xml).values()) == ['aab']


def test_selected_lines(pipelines):
    xml = page_xml(text_region('r1', (10, 30, 250, 120), text_line('l1', box(0)) + text_line('l2', box(1))))
    result = run(pipelines['full'], xml, target='TEXT_LINE', lines=['l2'])
    assert texts(result.xml) == {'l1': None, 'l2': 'bba'}


def test_unknown_selection_fails(pipelines):
    with pytest.raises(PageInputError, match='not part of the PAGE XML'):
        run(pipelines['full'], ONE_LINE, target='REGION', regions=['gone'])


def test_nothing_found_is_skipped(pipelines):
    blank = np.full((400, 600, 3), 255, dtype=np.uint8)
    with pytest.raises(PageSkipped, match='No text found'):
        run(pipelines['full'], image=blank)
    with pytest.raises(PageSkipped, match='No text found'):
        run(pipelines['full'], ONE_LINE, image=blank, segmentation='regions')
    with pytest.raises(PageSkipped, match='No text lines'):
        run(pipelines['full'], EMPTY_REGIONS, image=blank)


def test_lines_without_findings_keep_the_existing_ones(pipelines):
    blank = np.full((400, 600, 3), 255, dtype=np.uint8)
    result = run(pipelines['full'], ONE_LINE, image=blank, segmentation='lines')
    assert list(texts(result.xml)) == ['l1']
    assert result.warnings == ['No lines found in 2 text region(s): r1, r2']


def test_complete_segmentation_without_recognition_is_skipped(pipelines):
    xml = page_xml(text_region('r1', (10, 30, 250, 120), text_line('l1', box(0))))
    with pytest.raises(PageSkipped, match='already have lines'):
        run(pipelines['full'], xml, recognition=False)


def test_min_confidence_drops_only_new_lines(pipelines):
    with pytest.raises(PageSkipped, match='minimum confidence'):
        run(pipelines['full'], min_confidence=1.0)
    result = run(pipelines['full'], ONE_LINE, min_confidence=1.0)
    assert list(texts(result.xml)) == ['l1']
    result = run(pipelines['full'], ONE_LINE, segmentation='lines', min_confidence=1.0)
    assert list(texts(result.xml)) == ['l1']


def test_unusable_lines_are_reported(pipelines):
    broken = '<TextLine id="l9"><Coords points="0,0 10,0 20,0"/></TextLine>'
    xml = page_xml(text_region('r1', (10, 30, 250, 120), text_line('l1', box(0)) + broken))
    result = run(pipelines['full'], xml)
    assert texts(result.xml) == {'l1': 'ab', 'l9': None}
    assert result.warnings == ['1 line(s) have unusable geometry: l9']


def test_regions_without_outline_are_reported(pipelines):
    xml = page_xml(
        '<TextRegion id="r9"><Coords points="0,0 10,0 20,0"/></TextRegion>' + text_region('r2', (320, 270, 500, 320))
    )
    result = run(pipelines['full'], xml)
    assert found(result) == ['aab']
    assert result.warnings == ['1 text region(s) have no usable outline']


def test_scaled_images_are_mapped_to_the_page(pipelines):
    xml = page_xml(
        text_region('r1', (20, 60, 500, 240), text_line('l1', tuple(2 * value for value in box(0))))
        + text_region('r2', (640, 540, 1000, 640)),
        size=(1200, 800),
    )
    result = run(pipelines['full'], xml)
    assert found(result) == ['aab', 'ab']
    root = etree.fromstring(result.xml)
    coordinates = root.find('.//{*}TextLine[@id="r2-l1"]/{*}Coords').get('points')
    left = min(int(pair.split(',')[0]) for pair in coordinates.split())
    assert 640 <= left <= 680


def test_mismatching_images_fail(pipelines):
    xml = page_xml(text_region('r1', (0, 0, 100, 100)), size=(400, 600))
    with pytest.raises(PageInputError, match='does not fit the PAGE XML size 400x600'):
        run(pipelines['full'], xml)


def test_missing_page_size_is_filled_in(pipelines):
    xml = page_xml(text_region('r1', (10, 30, 250, 120), text_line('l1', box(0))), size=(0, 0))
    page = etree.fromstring(run(pipelines['full'], xml).xml).find('{*}Page')
    assert (page.get('imageWidth'), page.get('imageHeight')) == ('600', '400')


def test_missing_stages_fail(pipelines):
    with pytest.raises(PageInputError, match='no line detection'):
        run(pipelines['recognition-only'])
    with pytest.raises(PageInputError, match='no text recognition'):
        run(pipelines['layout-only'])


def test_failing_region_sorter_keeps_the_regions(pipelines, monkeypatch):
    sorter = pipelines['full'].layout.parsers[-1]

    def fail(*_args):
        raise ValueError('broken')

    monkeypatch.setattr(sorter, 'process_page', fail)
    result = run(pipelines['full'])
    assert found(result) == ['aab', 'ab', 'bba']
    assert 'reading order could not be determined' in result.warnings[0]


def test_only_needed_stages_are_loaded(model_dir):
    engine = Engine('full', model_dir / 'full' / 'config.ini')
    pipeline = Pipeline(engine, frozenset({'recognition'}), device='cpu')
    assert pipeline.layout is None
    assert pipeline.recognizer is not None
    assert multiprocessing.active_children() == []


def test_missing_model_files_are_explained(model_dir, tmp_path):
    (tmp_path / 'config.ini').write_text((model_dir / 'full' / 'config.ini').read_text())
    with pytest.raises(EngineError, match='cannot be loaded on cpu'):
        Pipeline(Engine('broken', tmp_path / 'config.ini'), frozenset({'layout'}), device='cpu')


def test_unavailable_devices_are_rejected():
    if torch.cuda.is_available():
        pytest.skip('CUDA is available')
    assert resolve_device('auto').type == 'cpu'
    with pytest.raises(EngineError, match='no CUDA device'):
        resolve_device('cuda:0')
