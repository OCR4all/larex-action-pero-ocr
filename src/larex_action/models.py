# SPDX-License-Identifier: MIT
import argparse
import hashlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import httpx

ENGINE_NAME = 'pero_eu_cz_print_newspapers_2022-09-26'
ARCHIVE_URL = 'https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ/download'
ARCHIVE_SHA256 = 'cb38dd0792c7145b8ba4dd64df255980e4633f4ca1d406f7f230b9c174c0c70d'
ARCHIVE_SIZE = 363_408_419
ENGINE_FILES = {
    'config.ini',
    'ocr_engine.json',
    'OCR_350000.pt',
    'OCR_350000.pt.cpu',
    'ParseNet_296000.pt',
    'ParseNet_296000.pt.cpu',
}
DOWNLOAD_ATTEMPTS = 3
CHUNK_SIZE = 1 << 20


class ModelDownloadError(RuntimeError):
    pass


def is_installed(directory: Path) -> bool:
    return all((directory / ENGINE_NAME / name).is_file() for name in ENGINE_FILES)


def download_engine(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=directory) as temp:
        archive = Path(temp) / 'engine.zip'
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                _download(archive)
                break
            except (httpx.HTTPError, ModelDownloadError) as error:
                print(f'Attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed: {error}', file=sys.stderr)
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise ModelDownloadError(f'Could not download {ARCHIVE_URL}: {error}') from error
        install_engine(archive, directory)


def install_engine(archive: Path, directory: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        members = {PurePosixPath(name).name: name for name in source.namelist() if not name.endswith('/')}
        if missing := ENGINE_FILES - members.keys():
            raise ModelDownloadError(f'The archive does not contain {", ".join(sorted(missing))}')
        target = directory / ENGINE_NAME
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(ENGINE_FILES):
            temporary = archive.with_name(name)
            with source.open(members[name]) as content, temporary.open('wb') as output:
                while chunk := content.read(CHUNK_SIZE):
                    output.write(chunk)
            temporary.replace(target / name)
    print(f'Installed {target}', flush=True)


def _download(target: Path) -> None:
    digest = hashlib.sha256()
    received = 0
    reported = 0
    with (
        httpx.stream('GET', ARCHIVE_URL, follow_redirects=True, timeout=60) as response,
        target.open('wb') as output,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes(CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            percent = received * 100 // ARCHIVE_SIZE
            if percent >= reported + 10:
                reported = percent - percent % 10
                print(f'Downloaded {reported}%', flush=True)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ModelDownloadError(f'{ARCHIVE_URL} does not match the pinned checksum')


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m larex_action.models')
    parser.add_argument('command', choices=['download'])
    parser.add_argument('--directory', type=Path, default=Path(os.environ.get('PERO_MODEL_DIR') or '/models'))
    directory = parser.parse_args(arguments).directory
    try:
        if not is_installed(directory):
            download_engine(directory)
    except (ModelDownloadError, OSError, zipfile.BadZipFile) as error:
        print(error, file=sys.stderr)
        return 1
    print(f'{ENGINE_NAME} is installed in {directory}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
