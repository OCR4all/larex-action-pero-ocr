# SPDX-License-Identifier: MIT
import hashlib
import zipfile

import httpx
import pytest

from larex_action import models


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as target:
        for name in models.ENGINE_FILES:
            target.writestr(f'{models.ENGINE_NAME}/{name}', name)
        target.writestr('../outside.txt', 'x')
    return path


@pytest.fixture
def served(archive, monkeypatch):
    content = archive.read_bytes()
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=content))
    monkeypatch.setattr(
        models.httpx, 'stream', lambda method, url, **_options: httpx.Client(transport=transport).stream(method, url)
    )
    monkeypatch.setattr(models, 'ARCHIVE_SHA256', hashlib.sha256(content).hexdigest())
    return content


def test_download_installs_the_engine(served, tmp_path):
    target = tmp_path / 'models'
    assert models.main(['download', '--directory', str(target)]) == 0
    assert {path.name for path in (target / models.ENGINE_NAME).iterdir()} == models.ENGINE_FILES
    assert [path.name for path in target.iterdir()] == [models.ENGINE_NAME]
    assert not (tmp_path / 'outside.txt').exists()


def test_installed_engines_are_not_downloaded_again(served, tmp_path, monkeypatch):
    models.main(['download', '--directory', str(tmp_path / 'models')])
    monkeypatch.setattr(models.httpx, 'stream', None)
    assert models.main(['download', '--directory', str(tmp_path / 'models')]) == 0


def test_changed_archives_are_refused(served, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(models, 'ARCHIVE_SHA256', '0' * 64)
    assert models.main(['download', '--directory', str(tmp_path / 'models')]) == 1
    assert 'does not match the pinned checksum' in capsys.readouterr().err
    assert list((tmp_path / 'models').iterdir()) == []


def test_incomplete_archives_are_refused(tmp_path):
    path = tmp_path / 'source.zip'
    with zipfile.ZipFile(path, 'w') as target:
        target.writestr('config.ini', '')
    with pytest.raises(models.ModelDownloadError, match='does not contain OCR_350000.pt'):
        models.install_engine(path, tmp_path / 'models')
