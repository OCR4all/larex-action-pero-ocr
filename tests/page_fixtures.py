# SPDX-License-Identifier: MIT
from functools import cache
from pathlib import Path

from lxml import etree

NAMESPACE = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2024-07-15'


@cache
def schema() -> etree.XMLSchema:
    return etree.XMLSchema(etree.parse(Path(__file__).parent / 'fixtures' / 'pagecontent-2024-07-15.xsd'))


def assert_valid(content: bytes) -> etree._Element:
    root = etree.fromstring(content)
    page_schema = schema()
    assert page_schema.validate(root), page_schema.error_log
    return root


def page_xml(body: str, size: tuple[int, int] = (600, 400), namespace: str = NAMESPACE) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{namespace}">
  <Metadata>
    <Creator>test</Creator>
    <Created>2020-01-01T00:00:00</Created>
    <LastChange>2020-01-01T00:00:00</LastChange>
  </Metadata>
  <Page imageFilename="page.png" imageWidth="{size[0]}" imageHeight="{size[1]}">{body}</Page>
</PcGts>""".encode()


def points(left: int, top: int, right: int, bottom: int) -> str:
    return f'{left},{top} {right},{top} {right},{bottom} {left},{bottom}'


def text_line(line_id: str, box: tuple[int, int, int, int], *, baseline: bool = True, extra: str = '') -> str:
    drawn = f'<Baseline points="{box[0]},{box[3] - 4} {box[2]},{box[3] - 4}"/>' if baseline else ''
    return f'<TextLine id="{line_id}"><Coords points="{points(*box)}"/>{drawn}{extra}</TextLine>'


def text_region(region_id: str, box: tuple[int, int, int, int], body: str = '') -> str:
    return f'<TextRegion id="{region_id}"><Coords points="{points(*box)}"/>{body}</TextRegion>'


def texts(content: bytes, index: int = 1) -> dict[str, str]:
    root = etree.fromstring(content)
    return {
        line.get('id'): line.findtext(f'{{*}}TextEquiv[@index="{index}"]/{{*}}Unicode')
        for line in root.iter('{*}TextLine')
    }
