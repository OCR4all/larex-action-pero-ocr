# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import get_args

import pytest
import yaml

from larex_action.config import Settings
from larex_action.jobs import Segmentation, Target
from larex_action.runner import RunParameters

ROOT = Path(__file__).parents[1]
DEFINITION = yaml.safe_load((ROOT / 'action.yml').read_text())


def test_parameters_match_the_processor():
    fields = {field.alias or name: field for name, field in RunParameters.model_fields.items()}
    assert set(DEFINITION['parameters']) == set(fields)
    for name, parameter in DEFINITION['parameters'].items():
        assert parameter.get('default', '') == fields[name].default


def test_segmentation_choices_match_the_processor():
    values = DEFINITION['parameters']['segmentation']['allowedValues']['values']
    assert [choice['value'] for choice in values] == list(get_args(Segmentation))


def test_targets_match_the_processor():
    assert DEFINITION['targets'] == list(get_args(Target))
    assert set(DEFINITION['outputs']['xml']['overwrites']) == set(get_args(Target))


@pytest.mark.parametrize('name', ['engines'])
def test_value_providers_exist(name):
    assert DEFINITION['parameters']['engine']['allowedValues']['provider'] == name


def test_environment_template_matches_the_settings(monkeypatch):
    template = dict(line.split('=', 1) for line in (ROOT / 'template.env').read_text().splitlines() if '=' in line)
    compose_only = {'NAME', 'PORT', 'MODEL_DIR'}
    assert set(template) - compose_only == set(Settings.model_fields) - {'PERO_MODEL_DIR'}
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    for name, value in template.items():
        monkeypatch.setenv(name, value or ('secret' if name == 'LAREX_DISPATCH_HMAC_SECRET' else ''))
    settings = Settings()
    for name, field in Settings.model_fields.items():
        if not field.is_required() and name not in ('LAREX_ROUTE_PREFIXES', 'PERO_MODEL_DIR'):
            assert getattr(settings, name) == field.default
