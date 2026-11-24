# SPDX-License-Identifier: MIT
import pytest

from larex_action.config import env
from larex_action.engines import (
    EngineError,
    discover_engines,
    engine_choices,
    engine_stages,
    find_engine,
)


def write(path, content='[PAGE_PARSER]\nRUN_OCR = yes\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(env, 'PERO_MODEL_DIR', tmp_path)
    monkeypatch.setattr(env, 'PERO_DEFAULT_ENGINE', None)
    return tmp_path


def test_discovers_directories_with_a_configuration(root):
    write(root / 'b' / 'config.ini')
    write(root / 'b' / 'config_cpu.ini')
    write(root / 'A' / 'config_cpu.ini')
    write(root / 'c' / 'newspapers.ini')
    write(root / 'ambiguous' / 'one.ini')
    write(root / 'ambiguous' / 'two.ini')
    write(root / '.hidden' / 'config.ini')
    write(root / 'empty' / 'readme.txt')
    write(root / 'config.ini')
    found = {engine.name: engine.config_file.name for engine in discover_engines()}
    assert found == {'A': 'config_cpu.ini', 'b': 'config.ini', 'c': 'newspapers.ini'}


def test_missing_model_directory_has_no_engines(root, monkeypatch):
    monkeypatch.setattr(env, 'PERO_MODEL_DIR', root / 'missing')
    assert engine_choices() == []
    with pytest.raises(EngineError, match='No pero-ocr engine found'):
        find_engine('')


def test_default_engine(root, monkeypatch):
    write(root / 'a' / 'config.ini')
    write(root / 'b' / 'config.ini')
    assert find_engine('').name == 'a'
    monkeypatch.setattr(env, 'PERO_DEFAULT_ENGINE', 'b')
    assert find_engine('').name == 'b'
    assert [(choice.value, choice.default) for choice in engine_choices()] == [('a', False), ('b', True)]


def test_unavailable_engines_are_rejected(root, monkeypatch):
    write(root / 'a' / 'config.ini')
    with pytest.raises(EngineError, match='"../a" is not available'):
        find_engine('../a')
    monkeypatch.setattr(env, 'PERO_DEFAULT_ENGINE', 'gone')
    with pytest.raises(EngineError, match='"gone" is not available'):
        find_engine('')


def test_stages_follow_the_configuration(installed_engines):
    assert engine_stages(find_engine('full')) == {'layout', 'recognition'}
    assert engine_stages(find_engine('layout-only')) == {'layout'}
    assert engine_stages(find_engine('recognition-only')) == {'recognition'}


@pytest.mark.parametrize('content', ['RUN_OCR = yes', '[OCR]\nx = 1', '[PAGE_PARSER]\nRUN_OCR = yes\nRUN_OCR = no'])
def test_broken_configurations_are_rejected(root, content):
    write(root / 'a' / 'config.ini', content)
    with pytest.raises(EngineError, match='Engine "a"'):
        find_engine('a').read_config()


def test_configured_cpu_use_is_ignored(installed_engines):
    config = find_engine('full').read_config()
    assert not config['OCR'].getboolean('USE_CPU')
    assert not config['LAYOUT_PARSER_1'].getboolean('USE_CPU')


def test_fingerprint_changes_with_the_files(root):
    write(root / 'a' / 'config.ini')
    before = find_engine('a').fingerprint
    assert find_engine('a').fingerprint == before
    write(root / 'a' / 'config.ini', '[PAGE_PARSER]\nRUN_OCR = no\n')
    assert find_engine('a').fingerprint != before
