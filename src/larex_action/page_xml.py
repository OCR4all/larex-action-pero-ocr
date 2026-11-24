# SPDX-License-Identifier: MIT
from collections.abc import Collection, Iterable, Sequence
from datetime import UTC, datetime

import numpy as np
from lxml import etree

from .errors import PageInputError
from .geometry import has_area, page_baseline, page_polygon
from .jobs import Target
from .model import Line, Region

NAMESPACE = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2024-07-15'
SCHEMA_INSTANCE = 'http://www.w3.org/2001/XMLSchema-instance'

BEFORE_LINE = {'AlternativeImage', 'Coords', 'UserDefined', 'Labels', 'Roles', 'TextLine'}
BEFORE_LINE_TEXT = {'AlternativeImage', 'Coords', 'Baseline', 'Word', 'TextEquiv'}
BEFORE_READING_ORDER = {'AlternativeImage', 'Border', 'PrintSpace'}
BEFORE_REGION = BEFORE_READING_ORDER | {'UserDefined', 'Labels', 'TextStyle', 'ReadingOrder', 'Layers', 'Relations'}

GROUPS = {'OrderedGroup', 'OrderedGroupIndexed', 'UnorderedGroup', 'UnorderedGroupIndexed'}
EMPTY_REMOVABLE = GROUPS | {'Layer', 'Layers', 'Relations', 'ReadingOrder'}


def local_name(element: etree._Element) -> str:
    return etree.QName(element).localname if isinstance(element.tag, str) else ''


def children(element: etree._Element, name: str) -> list[etree._Element]:
    return [child for child in element if local_name(child) == name]


def first_child(element: etree._Element, name: str) -> etree._Element | None:
    return next((child for child in element if local_name(child) == name), None)


def is_region(element: etree._Element) -> bool:
    return local_name(element).endswith('Region')


def is_nested(parent: etree._Element, child: etree._Element) -> bool:
    return is_region(parent) and is_region(child)


def parse_points(element: etree._Element | None) -> np.ndarray | None:
    if element is None:
        return None
    points = []
    for pair in (element.get('points') or '').split():
        x, _, y = pair.partition(',')
        try:
            points.append((float(x), float(y)))
        except ValueError:
            return None
    if not points:
        for point in children(element, 'Point'):
            try:
                points.append((float(point.get('x', '')), float(point.get('y', ''))))
            except ValueError:
                return None
    result = np.asarray(points, dtype=float)
    return result if len(points) >= 2 and np.isfinite(result).all() else None


def parse_polygon(element: etree._Element | None) -> np.ndarray | None:
    points = parse_points(element)
    return points if points is not None and has_area(points) else None


def format_points(points: np.ndarray) -> str:
    return ' '.join(f'{x},{y}' for x, y in points)


def equiv_index(equiv: etree._Element) -> int | None:
    try:
        return int(equiv.get('index', ''))
    except ValueError:
        return None


def _positive_int(value: str | None) -> int:
    try:
        return max(int(value or ''), 0)
    except ValueError:
        return 0


class PageDocument:
    def __init__(self, root: etree._Element) -> None:
        self.root = root
        page = next((element for element in root.iter() if local_name(element) == 'Page'), None)
        if page is None:
            raise PageInputError('Input does not contain a PAGE XML Page element')
        self.page = page
        self.namespace = etree.QName(page).namespace
        self._ids = {element.get('id') for element in root.iter() if element.get('id')}

    @classmethod
    def parse(cls, content: bytes) -> 'PageDocument':
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=True)
        try:
            return cls(etree.fromstring(content, parser=parser))
        except etree.XMLSyntaxError as exc:
            raise PageInputError(f'Invalid PAGE XML: {exc}') from exc

    @classmethod
    def create(cls, image_name: str, size: tuple[int, int], creator: str) -> 'PageDocument':
        root = etree.Element(
            f'{{{NAMESPACE}}}PcGts',
            {f'{{{SCHEMA_INSTANCE}}}schemaLocation': f'{NAMESPACE} {NAMESPACE}/pagecontent.xsd'},
            nsmap={None: NAMESPACE, 'xsi': SCHEMA_INSTANCE},
        )
        now = datetime.now(UTC).replace(microsecond=0, tzinfo=None).isoformat()
        metadata = etree.SubElement(root, f'{{{NAMESPACE}}}Metadata')
        for name, value in (('Creator', creator), ('Created', now), ('LastChange', now)):
            etree.SubElement(metadata, f'{{{NAMESPACE}}}{name}').text = value
        etree.SubElement(
            root,
            f'{{{NAMESPACE}}}Page',
            imageFilename=image_name,
            imageWidth=str(size[0]),
            imageHeight=str(size[1]),
        )
        return cls(root)

    def serialize(self) -> bytes:
        return etree.tostring(self.root.getroottree(), xml_declaration=True, encoding='UTF-8')

    @property
    def size(self) -> tuple[int, int]:
        return _positive_int(self.page.get('imageWidth')), _positive_int(self.page.get('imageHeight'))

    @size.setter
    def size(self, size: tuple[int, int]) -> None:
        self.page.set('imageWidth', str(size[0]))
        self.page.set('imageHeight', str(size[1]))

    def text_regions(self) -> list[etree._Element]:
        return [element for element in self.page.iter() if local_name(element) == 'TextRegion']

    def regions_in_scope(
        self, target: Target, *, region_ids: Collection[str] = (), line_ids: Collection[str] = ()
    ) -> list[Region]:
        result = []
        for element in self.text_regions():
            lines = children(element, 'TextLine')
            if target == 'REGION' and not any(
                ancestor.get('id') in region_ids for ancestor in [element, *element.iterancestors()]
            ):
                continue
            if target == 'TEXT_LINE':
                lines = [line for line in lines if line.get('id') in line_ids]
                if not lines:
                    continue
            result.append(
                Region(
                    polygon=parse_polygon(first_child(element, 'Coords')),
                    element=element,
                    lines=[self._read_line(line) for line in lines],
                )
            )
        return result

    @staticmethod
    def _read_line(element: etree._Element) -> Line:
        return Line(
            baseline=parse_points(first_child(element, 'Baseline')),
            polygon=parse_polygon(first_child(element, 'Coords')),
            element=element,
        )

    def replace_text_regions(self, regions: Sequence[Region]) -> None:
        previous = children(self.page, 'TextRegion')
        removed = {item.get('id') for element in previous for item in element.iter() if item.get('id')}
        kept = [region_id for region_id in self._reading_order_ids() if region_id not in removed]
        for element in previous:
            self.page.remove(element)
        self._drop_references(removed)

        position = self._insert_position(self.page, BEFORE_REGION)
        for region in regions:
            region.element = self._build_region(region)
            if region.element is not None:
                self.page.insert(position, region.element)
                position += 1
        self._write_reading_order([region.element.get('id') for region in regions if region.element is not None] + kept)

    def replace_lines(self, region: Region) -> None:
        for element in children(region.element, 'TextLine'):
            removed = {item.get('id') for item in element.iter() if item.get('id')}
            region.element.remove(element)
            self._drop_references(removed)
        position = self._insert_position(region.element, BEFORE_LINE)
        for line in region.lines:
            line.element = self._build_line(line, region.element.get('id') or 'r')
            if line.element is not None:
                region.element.insert(position, line.element)
                position += 1

    def write_text(self, line: Line, index: int) -> None:
        if line.element is None or line.text is None:
            return
        self._set_text(line.element, line.text, index, line.confidence, BEFORE_LINE_TEXT)

    def refresh_region_text(self, region: Region, index: int) -> None:
        if region.element is None:
            return
        has_text = any(equiv_index(equiv) == index for equiv in children(region.element, 'TextEquiv'))
        if not has_text:
            return
        texts = []
        for line in children(region.element, 'TextLine'):
            equiv = next((item for item in children(line, 'TextEquiv') if equiv_index(item) == index), None)
            content = first_child(equiv, 'Unicode') if equiv is not None else None
            if content is not None and content.text:
                texts.append(content.text)
        self._set_text(region.element, '\n'.join(texts), index, None, None)

    def touch(self) -> None:
        metadata = first_child(self.root, 'Metadata')
        changed = first_child(metadata, 'LastChange') if metadata is not None else None
        if changed is not None:
            changed.text = datetime.now(UTC).replace(microsecond=0, tzinfo=None).isoformat()

    def _tag(self, name: str) -> str:
        return f'{{{self.namespace}}}{name}' if self.namespace else name

    def _new_id(self, stem: str) -> str:
        number = 1
        while f'{stem}{number}' in self._ids:
            number += 1
        element_id = f'{stem}{number}'
        self._ids.add(element_id)
        return element_id

    @staticmethod
    def _insert_position(parent: etree._Element, preceding: Iterable[str]) -> int:
        names = set(preceding)
        return max(
            (index + 1 for index, child in enumerate(parent) if local_name(child) in names or is_nested(parent, child)),
            default=0,
        )

    def _build_region(self, region: Region) -> etree._Element | None:
        polygon = page_polygon(region.polygon, self.size) if region.polygon is not None else None
        if polygon is None:
            return None
        element = etree.Element(self._tag('TextRegion'), id=self._new_id('r'))
        etree.SubElement(element, self._tag('Coords'), points=format_points(polygon))
        for line in region.lines:
            line.element = self._build_line(line, element.get('id') or 'r')
            if line.element is not None:
                element.append(line.element)
        return element

    def _build_line(self, line: Line, region_id: str) -> etree._Element | None:
        polygon = page_polygon(line.polygon, self.size) if line.polygon is not None else None
        baseline = page_baseline(line.baseline, self.size) if line.baseline is not None else None
        if polygon is None or baseline is None:
            return None
        element = etree.Element(self._tag('TextLine'), id=self._new_id(f'{region_id}-l'))
        etree.SubElement(element, self._tag('Coords'), points=format_points(polygon))
        etree.SubElement(element, self._tag('Baseline'), points=format_points(baseline))
        return element

    def _set_text(
        self,
        element: etree._Element,
        text: str,
        index: int,
        confidence: float | None,
        preceding: Collection[str] | None,
    ) -> None:
        equivs = children(element, 'TextEquiv')
        for equiv in equivs:
            if equiv_index(equiv) == index:
                element.remove(equiv)
        if not text:
            return
        equiv = etree.Element(self._tag('TextEquiv'), index=str(index))
        if confidence is not None:
            equiv.set('conf', f'{min(max(confidence, 0.0), 1.0):.4f}')
        etree.SubElement(equiv, self._tag('Unicode')).text = text
        following = next(
            (
                item
                for item in children(element, 'TextEquiv')
                if (other := equiv_index(item)) is not None and other > index
            ),
            None,
        )
        if following is not None:
            following.addprevious(equiv)
        elif preceding is not None:
            element.insert(self._insert_position(element, preceding), equiv)
        elif (style := first_child(element, 'TextStyle')) is not None:
            style.addprevious(equiv)
        else:
            element.append(equiv)

    def _reading_order_ids(self) -> list[str]:
        order = first_child(self.page, 'ReadingOrder')
        if order is None:
            return []
        return list(dict.fromkeys(item.get('regionRef') for item in order.iter() if item.get('regionRef')))

    def _write_reading_order(self, region_ids: Sequence[str]) -> None:
        existing = first_child(self.page, 'ReadingOrder')
        if existing is not None:
            self.page.remove(existing)
        if not region_ids:
            return
        order = etree.Element(self._tag('ReadingOrder'))
        group = etree.SubElement(order, self._tag('OrderedGroup'), id=self._new_id('ro'))
        for index, region_id in enumerate(region_ids):
            etree.SubElement(group, self._tag('RegionRefIndexed'), index=str(index), regionRef=region_id)
        self.page.insert(self._insert_position(self.page, BEFORE_READING_ORDER), order)

    def _drop_references(self, removed: Collection[str]) -> None:
        for item in [item for item in self.page.iter() if item.get('regionRef') in removed]:
            parent = item.getparent()
            if parent is None:
                continue
            if local_name(item) in GROUPS:
                del item.attrib['regionRef']
                continue
            if local_name(parent) == 'Relation':
                item, parent = parent, parent.getparent()
            while parent is not None:
                parent.remove(item)
                if len(parent) or local_name(parent) not in EMPTY_REMOVABLE:
                    break
                item, parent = parent, parent.getparent()
