# SPDX-License-Identifier: MIT
import pytest
from fake_larex import PROCESSOR_ID, FakeLarex, sign
from fastapi.testclient import TestClient
from page_fixtures import assert_valid, texts
from stub_engine import draw_page, encode

from larex_action import main
from larex_action.config import env

SECRET = 'test-dispatch-secret'


@pytest.fixture
def larex(monkeypatch):
    server = FakeLarex()
    from larex_actions import fastapi

    original = fastapi.ActionClient.from_dispatch

    def from_dispatch(payload, **options):
        return original(payload, **{**options, 'allowed_callback_origins': {server.origin}})

    monkeypatch.setattr(fastapi.ActionClient, 'from_dispatch', from_dispatch)
    yield server
    server.close()


@pytest.fixture
def client(installed_engines):
    with TestClient(main.app) as test_client:
        yield test_client


def post(client, path, payload, secret=SECRET):
    body, headers = sign(payload, path, secret)
    return client.post(path, content=body, headers=headers)


@pytest.mark.parametrize('prefix', ['', '/pero-ocr'])
def test_endpoints(client, prefix):
    assert client.get(f'{prefix}/health').json() == {'status': 'ok'}
    request = {'protocolVersion': 1, 'requestId': 'request-1', 'processorId': PROCESSOR_ID, 'capabilities': {}}
    response = post(client, f'{prefix}/preflight', request)
    assert response.status_code == 200
    assert response.json()['capabilities']['parameterValueDiscovery']


def test_engines_are_offered_as_parameter_values(client, monkeypatch):
    monkeypatch.setattr(env, 'PERO_DEFAULT_ENGINE', 'layout-only')
    request = {
        'protocolVersion': 1,
        'requestId': 'request-1',
        'processorId': PROCESSOR_ID,
        'providers': ['engines'],
        'capabilities': {'parameterChoiceDefaults': True},
    }
    choices = post(client, '/parameter-values', request).json()['values']['engines']
    assert [(choice['value'], choice.get('default', False)) for choice in choices] == [
        ('full', False),
        ('layout-only', True),
        ('recognition-only', False),
    ]


def test_dispatches_with_a_wrong_signature_are_refused(client, larex):
    payload = larex.prepare([(encode(draw_page([('ab', 20, 60)])), None)])
    assert post(client, '/dispatch', payload, secret='wrong').status_code == 401
    assert larex.results == []


def test_signed_run_with_a_real_worker(client, larex, capfd):
    pages = [
        (encode(draw_page([('ab', 20, 60), ('bba', 20, 100)])), None),
        (b'not an image', None),
        (encode(draw_page([])), None),
        (encode(draw_page([('aab', 100, 200)]), '.tif'), None),
    ]
    response = post(client, '/dispatch', larex.prepare(pages, engine='full'))
    assert response.status_code == 200
    assert larex.finished.wait(60)

    *page_results, final = larex.results
    assert [manifest['pageOutcome']['status'] for manifest, _ in page_results] == [
        'SUCCEEDED',
        'FAILED',
        'SKIPPED',
        'SUCCEEDED',
    ]
    assert 'Unreadable page image' in page_results[1][0]['pageOutcome']['message']
    first, last = (next(iter(files.values())) for _, files in (page_results[0], page_results[3]))
    assert_valid(first)
    assert sorted(texts(first).values()) == ['ab', 'bba']
    assert list(texts(last).values()) == ['aab']
    assert final[0]['status'] == 'completed'
    assert final[0]['message'] == 'Processed 2/4 page(s) with full'
    assert larex.heartbeats
    assert 'Loaded engine full on cpu' in capfd.readouterr().err
