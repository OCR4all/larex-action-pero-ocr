# SPDX-License-Identifier: MIT
import os
from pathlib import Path

import pytest

os.environ.update(
    {
        'LAREX_PROCESSOR_ID': 'pero-ocr',
        'LAREX_DISPATCH_HMAC_SECRET': 'test-dispatch-secret',
        'LAREX_ALLOWED_CALLBACK_ORIGINS': '["http://127.0.0.1"]',
        'LAREX_ROUTE_PREFIXES': '["/pero-ocr"]',
        'PERO_CPU_THREADS': '1',
        'PERO_DEVICE': 'cpu',
    }
)


@pytest.fixture(scope='session')
def model_dir(tmp_path_factory) -> Path:
    from stub_engine import write_engine

    root = tmp_path_factory.mktemp('models')
    write_engine(root / 'full')
    write_engine(root / 'layout-only', recognition=False)
    write_engine(root / 'recognition-only', layout=False)
    return root


@pytest.fixture
def installed_engines(model_dir, monkeypatch) -> Path:
    from larex_action.config import env

    monkeypatch.setattr(env, 'PERO_MODEL_DIR', model_dir)
    monkeypatch.setattr(env, 'PERO_DEFAULT_ENGINE', None)
    monkeypatch.setenv('PERO_MODEL_DIR', str(model_dir))
    return model_dir
