# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from lxml import etree
from page_fixtures import assert_valid, page_xml, points, text_line, text_region, texts

from larex_action.errors import PageInputError
from larex_action.model import Line, Region
from larex_action.page_xml import PageDocument, parse_points

LEGACY = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'


def new_line(left, top, right, bottom, text=None, confidence=None):
    return Line(
        baseline=np.array([[left, bottom - 4], [right, bottom - 4]], dtype=float),
        polygon=np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float),
        text=text,
        confidence=confidence,
    )


def new_region(left, top, right, bottom, *lines):
    return Region(
        polygon=np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float), lines=list(lines)
    )


def order_of(content):
    return [item.get('regionRef') for item in etree.fromstring(content).iter('{*}RegionRefIndexed')]


def test_creates_a_valid_document():
    document = PageDocument.create('page.png', (600, 400), 'pero-ocr 0.7.0')
    line = new_line(20, 20, 200, 40, 'text', 0.98765)
    document.replace_text_regions([new_region(10, 10, 210, 50, line), new_region(10, 60, 210, 90)])
    document.write_text(line, 1)
    root = assert_valid(document.serialize())
    assert [region.get('id') for region in root.iter('{*}TextRegion')] == ['r1', 'r2']
    assert order_of(document.serialize()) == ['r1', 'r2']
    equiv = root.find('.//{*}TextEquiv')
    assert (equiv.get('index'), equiv.get('conf')) == ('1', '0.9877')


def test_replacing_regions_keeps_other_content_and_references():
    body = f"""
    <AlternativeImage filename="bin.png"/>
    <ReadingOrder><OrderedGroup id="ro">
      <RegionRefIndexed index="0" regionRef="old"/>
      <RegionRefIndexed index="1" regionRef="image"/>
    </OrderedGroup></ReadingOrder>
    <Layers><Layer id="la1" zIndex="1"><RegionRef regionRef="old"/></Layer></Layers>
    <Relations>
      <Relation type="link"><SourceRegionRef regionRef="old"/><TargetRegionRef regionRef="image"/></Relation>
    </Relations>
    {text_region('old', (0, 0, 100, 100), text_line('l1', (0, 0, 90, 20)))}
    <ImageRegion id="image"><Coords points="{points(300, 0, 400, 100)}"/></ImageRegion>"""
    document = PageDocument.parse(page_xml(body))
    document.replace_text_regions([new_region(10, 10, 210, 50, new_line(20, 20, 200, 40))])
    content = document.serialize()
    root = assert_valid(content)
    assert order_of(content) == ['r1', 'image']
    assert root.find('.//{*}Layers') is None
    assert root.find('.//{*}Relations') is None
    assert root.find('.//{*}ImageRegion') is not None
    assert root.find('.//{*}AlternativeImage') is not None
    assert root.find('.//*[@id="old"]') is None


def test_new_ids_do_not_collide():
    document = PageDocument.parse(page_xml(text_region('r1', (0, 0, 9, 9)) + text_region('r3', (0, 20, 9, 29))))
    document.replace_text_regions([new_region(0, 0, 90, 90), new_region(0, 100, 90, 190)])
    root = assert_valid(document.serialize())
    assert [region.get('id') for region in root.iter('{*}TextRegion')] == ['r2', 'r4']


def test_replacing_lines_keeps_the_element_order():
    body = text_region(
        'r1',
        (0, 0, 300, 100),
        text_line('old', (0, 0, 90, 20))
        + text_region('nested', (10, 10, 50, 50))
        + '<TextEquiv index="0"><Unicode>truth</Unicode></TextEquiv><TextStyle fontSize="10"/>',
    ).replace(text_line('old', (0, 0, 90, 20)) + text_region('nested', (10, 10, 50, 50)), '{swap}')
    body = body.replace('{swap}', text_region('nested', (10, 10, 50, 50)) + text_line('old', (0, 0, 90, 20)))
    document = PageDocument.parse(page_xml(body))
    region = document.regions_in_scope('PAGE')[0]
    region.lines = [new_line(0, 0, 200, 20), new_line(0, 30, 200, 50)]
    document.replace_lines(region)
    root = assert_valid(document.serialize())
    assert [line.get('id') for line in root.iter('{*}TextLine')] == ['r1-l1', 'r1-l2']


def test_text_is_stored_under_its_index():
    equivs = (
        '<TextEquiv index="0"><Unicode>truth</Unicode></TextEquiv>'
        '<TextEquiv index="1" conf="0.5"><Unicode>previous</Unicode></TextEquiv>'
        '<TextEquiv index="2"><Unicode>other</Unicode></TextEquiv>'
    )
    body = text_region(
        'r1', (0, 0, 300, 100), text_line('l1', (0, 0, 90, 20), extra=equivs + '<TextStyle bold="true"/>')
    )
    document = PageDocument.parse(page_xml(body))
    line = document.regions_in_scope('PAGE')[0].lines[0]
    line.text, line.confidence = 'new', 1.2
    document.write_text(line, 1)
    content = document.serialize()
    root = assert_valid(content)
    assert [equiv.get('index') for equiv in root.iter('{*}TextEquiv')] == ['0', '1', '2']
    assert (texts(content, 0), texts(content, 1), texts(content, 2)) == (
        {'l1': 'truth'},
        {'l1': 'new'},
        {'l1': 'other'},
    )
    assert root.find('.//{*}TextEquiv[@index="1"]').get('conf') == '1.0000'


def test_empty_text_removes_the_previous_result():
    equiv = '<TextEquiv index="1"><Unicode>previous</Unicode></TextEquiv>'
    document = PageDocument.parse(
        page_xml(text_region('r1', (0, 0, 300, 100), text_line('l1', (0, 0, 90, 20), extra=equiv)))
    )
    line = document.regions_in_scope('PAGE')[0].lines[0]
    line.text = ''
    document.write_text(line, 1)
    assert texts(document.serialize()) == {'l1': None}


def test_region_text_follows_its_lines():
    equiv = '<TextEquiv index="1"><Unicode>outdated</Unicode></TextEquiv>'
    body = text_region(
        'r1', (0, 0, 300, 100), text_line('l1', (0, 0, 90, 20)) + text_line('l2', (0, 30, 90, 50)) + equiv
    ) + text_region('r2', (0, 200, 300, 300), text_line('l3', (0, 200, 90, 220)))
    document = PageDocument.parse(page_xml(body))
    for region in document.regions_in_scope('PAGE'):
        for line in region.lines:
            line.text = f'text {line.id}'
            document.write_text(line, 1)
        document.refresh_region_text(region, 1)
    root = assert_valid(document.serialize())
    first, second = root.iter('{*}TextRegion')
    assert first.findtext('{*}TextEquiv/{*}Unicode') == 'text l1\ntext l2'
    assert second.find('{*}TextEquiv') is None


def test_scopes():
    body = text_region(
        'r1',
        (0, 0, 300, 100),
        text_region('r2', (0, 0, 100, 50), text_line('l2', (0, 0, 90, 20))) + text_line('l1', (0, 60, 90, 80)),
    ) + text_region('r3', (0, 200, 300, 300), text_line('l3', (0, 200, 90, 220)))
    document = PageDocument.parse(page_xml(body))

    def lines(*args, **kwargs):
        return [line.id for region in document.regions_in_scope(*args, **kwargs) for line in region.lines]

    assert lines('PAGE') == ['l1', 'l2', 'l3']
    assert lines('REGION', region_ids=['r1']) == ['l1', 'l2']
    assert lines('REGION', region_ids=['r2']) == ['l2']
    assert lines('TEXT_LINE', line_ids=['l3', 'l1']) == ['l1', 'l3']
    assert lines('TEXT_LINE', line_ids=['missing']) == []


def test_older_versions_and_point_elements():
    body = """<TextRegion id="r1"><Coords><Point x="0" y="0"/><Point x="90" y="0"/><Point x="90" y="20"/></Coords>
    <TextLine id="l1"><Coords><Point x="0" y="0"/><Point x="90" y="0"/><Point x="90" y="20"/></Coords></TextLine>
    </TextRegion>"""
    document = PageDocument.parse(page_xml(body, namespace=LEGACY))
    region = document.regions_in_scope('PAGE')[0]
    assert region.polygon.shape == (3, 2)
    region.lines[0].text = 'new'
    document.write_text(region.lines[0], 1)
    assert texts(document.serialize()) == {'l1': 'new'}
    assert f'{{{LEGACY}}}TextEquiv' in {element.tag for element in document.root.iter()}


@pytest.mark.parametrize('value', ['', '5,5', 'a,b c,d', '1,2 3', '0,0 nan,1 2,2', '0,0 inf,1 2,2'])
def test_unusable_points(value):
    assert parse_points(etree.Element('Coords', points=value)) is None


def test_flat_outlines_are_unusable():
    body = '<TextRegion id="r1"><Coords points="0,0 10,0 20,0"/></TextRegion>'
    assert PageDocument.parse(page_xml(body)).regions_in_scope('PAGE')[0].polygon is None


@pytest.mark.parametrize('content', [b'', b'<PcGts', b'<PcGts/>'])
def test_invalid_documents_are_rejected(content):
    with pytest.raises(PageInputError):
        PageDocument.parse(content)


def test_external_entities_are_not_resolved(tmp_path):
    secret = tmp_path / 'secret.txt'
    secret.write_text('secret')
    content = page_xml(text_region('r1', (0, 0, 9, 9), '<TextEquiv><Unicode>&leak;</Unicode></TextEquiv>')).replace(
        b'<PcGts', f'<!DOCTYPE PcGts [<!ENTITY leak SYSTEM "{secret.as_uri()}">]><PcGts'.encode(), 1
    )
    assert b'secret</' not in PageDocument.parse(content).serialize()
