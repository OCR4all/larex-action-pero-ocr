# SPDX-License-Identifier: MIT
import asyncio
import importlib
import re

import fake_worker
import pytest
from contexts import make_context
from larex_actions import ActionCancelled
from page_fixtures import page_xml

from larex_action import pool as pool_module
from larex_action import worker
from larex_action.config import Settings, env
from larex_action.pool import WorkerPool

runner = importlib.import_module('larex_action.runner')
EMPTY = page_xml('')


@pytest.fixture(autouse=True)
async def fake_pool(installed_engines, monkeypatch):
    monkeypatch.setattr(pool_module, 'worker', fake_worker.worker)
    pool = WorkerPool()
    monkeypatch.setattr(runner, 'pool', pool)
    monkeypatch.setattr(env, 'RUN_SEMAPHORE', asyncio.Semaphore(1), raising=False)
    yield pool
    await pool.close()


async def run(images, **kwargs):
    ctx = make_context(images, **kwargs)
    await runner.runner(ctx)
    return ctx


def worker_notes(ctx):
    return [re.search(rb'<!--(\d+) (.*)-->', page['files'][0].content).groups() for page in ctx.pages if page['files']]


async def test_pages_are_processed_in_order_by_one_worker():
    ctx = await run([b'ok'] * 5, xml=[EMPTY, None, EMPTY])
    assert ctx.failed is None
    assert ctx.completed == 'Processed 5/5 page(s) with full'
    assert [page['page_id'] for page in ctx.pages] == ['p1', 'p2', 'p3', 'p4', 'p5']
    assert ctx.statuses == ['SUCCEEDED'] * 5
    assert ctx.pages[0]['reason'] == '[1/5] Page 1: 4 line(s)'
    assert ctx.pages[0]['warnings'] == ['warned']
    stored = ctx.pages[0]['files'][0]
    assert (stored.type, stored.file_name, stored.page_id) == ('xml', 'Page 1.xml', 'p1')
    assert len({note[0] for note in worker_notes(ctx)}) == 1
    assert [b'True' in note[1] for note in worker_notes(ctx)] == [True, False, False, False, False]


async def test_failed_and_skipped_pages_do_not_stop_the_run():
    ctx = await run([b'ok', b'fail', b'skip', None, b'ok'])
    assert ctx.statuses == ['SUCCEEDED', 'FAILED', 'SKIPPED', 'FAILED', 'SUCCEEDED']
    assert ctx.pages[1]['reason'] == '[2/5] Page 2 failed: ValueError: bad page'
    assert ctx.pages[2]['reason'] == '[3/5] Page 3 skipped: No text found on the page'
    assert ctx.pages[2]['warnings'] == ['careful']
    assert ctx.pages[3]['reason'] == '[4/5] Page 4 failed: Page has no image'
    assert ctx.completed == 'Processed 2/5 page(s) with full'
    assert len({note[0] for note in worker_notes(ctx)}) == 1


@pytest.mark.parametrize('kind', [b'crash', b'restart'])
async def test_worker_is_replaced(kind):
    ctx = await run([b'ok', kind, b'ok'])
    assert ctx.statuses == ['SUCCEEDED', 'FAILED', 'SUCCEEDED']
    assert len({note[0] for note in worker_notes(ctx)}) == 2
    assert ctx.completed == 'Processed 2/3 page(s) with full'


async def test_repeated_crashes_fail_the_run():
    ctx = await run([b'crash'] * 5)
    assert ctx.statuses == ['FAILED'] * 3
    assert 'crashed on 3 pages in a row' in ctx.failed
    assert 'code 3' in ctx.failed


async def test_separated_crashes_do_not_fail_the_run():
    ctx = await run([b'crash', b'crash', b'ok', b'crash', b'crash', b'ok'])
    assert ctx.failed is None
    assert ctx.completed == 'Processed 2/6 page(s) with full'


async def test_page_timeout(monkeypatch):
    monkeypatch.setattr(env, 'ACTION_PROCESSING_TIMEOUT', 1)
    ctx = await run([b'hang', b'ok'])
    assert ctx.statuses == ['FAILED', 'SUCCEEDED']
    assert ctx.pages[0]['reason'] == '[1/2] Page 1 timed out after 1 seconds'


async def test_page_timeout_can_fail_the_run(monkeypatch):
    monkeypatch.setattr(env, 'ACTION_PROCESSING_TIMEOUT', 1)
    monkeypatch.setattr(env, 'ACTION_TIMEOUT_MODE', 'fail_action')
    ctx = await run([b'hang', b'ok'])
    assert ctx.pages == []
    assert ctx.failed == '[1/2] Page 1 timed out after 1 seconds'


@pytest.mark.parametrize(
    ('mode', 'message'),
    [
        ('error', 'Worker failed to load: EngineError: broken'),
        ('crash', 'Worker exited unexpectedly with code 7 while loading'),
        ('hang', 'Worker did not load within 2 seconds'),
    ],
)
async def test_loading_failures_fail_the_run(monkeypatch, mode, message):
    monkeypatch.setattr(env, 'ACTION_LOADING_TIMEOUT', 2 if mode == 'hang' else 60)
    monkeypatch.setenv('FAKE_WORKER_LOADING', mode)
    ctx = await run([b'ok'])
    assert ctx.failed == message
    assert ctx.pages == []


async def test_unknown_worker_messages_fail_the_run():
    ctx = await run([b'unknown'])
    assert 'unknown result type' in ctx.failed


async def test_cancellation_stops_the_worker(fake_pool):
    ctx = make_context([b'ok', b'hang', b'ok'])
    ctx.cancel_after_pages = 1
    with pytest.raises(ActionCancelled):
        await runner.runner(ctx)
    assert ctx.statuses == ['SUCCEEDED']
    assert ctx.completed is None
    assert fake_pool._idle == []


async def test_rejected_results_do_not_stop_the_run():
    ctx = make_context([b'ok', b'ok', b'ok'])
    ctx.rejected_pages = {'p2'}
    await runner.runner(ctx)
    assert [page['page_id'] for page in ctx.pages] == ['p1', 'p3']
    assert ctx.completed == 'Processed 2/3 page(s) with full'


async def test_download_errors_abort_the_run(fake_pool):
    ctx = make_context([b'ok', RuntimeError('gone')])
    with pytest.raises(RuntimeError, match='gone'):
        await runner.runner(ctx)
    assert fake_pool._idle == []


async def test_worker_is_reused_by_the_next_run():
    first = await run([b'ok'])
    second = await run([b'ok'])
    assert worker_notes(first)[0][0] == worker_notes(second)[0][0]
    assert b'True' in worker_notes(second)[0][1]


async def test_worker_is_not_reused_for_another_engine(fake_pool):
    first = await run([b'ok'], parameters={'engine': 'full'})
    second = await run([b'ok'], parameters={'engine': 'layout-only', 'recognition': False})
    assert worker_notes(first)[0][0] != worker_notes(second)[0][0]
    assert len(fake_pool._idle) == 1


async def test_worker_with_fewer_stages_is_replaced(fake_pool):
    first = await run([b'ok'], parameters={'segmentation': 'keep'})
    second = await run([b'ok'])
    third = await run([b'ok'], parameters={'segmentation': 'keep'})
    assert worker_notes(first)[0][0] != worker_notes(second)[0][0]
    assert worker_notes(second)[0][0] == worker_notes(third)[0][0]


async def test_changed_engine_files_load_a_new_worker(installed_engines):
    first = await run([b'ok'])
    config = installed_engines / 'full' / 'config.ini'
    original = config.read_text()
    config.write_text(original + '\n')
    try:
        second = await run([b'ok'])
    finally:
        config.write_text(original)
    assert worker_notes(first)[0][0] != worker_notes(second)[0][0]


async def test_dead_idle_workers_are_replaced(fake_pool):
    await run([b'ok'])
    fake_pool._idle[0].process.kill()
    fake_pool._idle[0].process.join()
    ctx = await run([b'ok'])
    assert ctx.statuses == ['SUCCEEDED']


async def test_idle_workers_expire(fake_pool, monkeypatch):
    monkeypatch.setattr(env, 'ACTION_IDLE_TIMEOUT', 1)
    await run([b'ok'])
    process = fake_pool._idle[0].process
    for _ in range(60):
        if not process.is_alive():
            break
        await asyncio.sleep(0.1)
    assert not process.is_alive()
    assert fake_pool._idle == []


async def test_keeping_workers_can_be_disabled(fake_pool, monkeypatch):
    monkeypatch.setattr(env, 'ACTION_IDLE_TIMEOUT', 0)
    ctx = await run([b'ok'])
    assert ctx.statuses == ['SUCCEEDED']
    assert fake_pool._idle == []


async def test_selection_is_passed_to_the_worker():
    target = {'type': 'REGION', 'pages': [{'pageId': 'p1', 'regionIds': ['r1', 'r2']}]}
    ctx = await run([b'ok', b'ok'], target=target, xml=[EMPTY, None])
    assert ctx.statuses == ['SUCCEEDED', 'FAILED']
    assert worker_notes(ctx)[0][1] == b"REGION ('r1', 'r2') () True"
    assert 'Page has no PAGE XML' in ctx.pages[1]['reason']


async def test_only_the_first_image_is_downloaded():
    ctx = make_context([b'ok', b'ok'])
    ctx.input.pages[0].images.append(ctx.input.pages[1].images[0])
    await runner.runner(ctx)
    assert ctx.downloads == ['img1', 'img2']


async def test_prefetching_can_be_disabled(monkeypatch):
    monkeypatch.setattr(env, 'PERO_PREFETCH_PAGES', 0)
    ctx = await run([b'ok', b'crash', b'ok'])
    assert ctx.statuses == ['SUCCEEDED', 'FAILED', 'SUCCEEDED']


async def test_empty_input_fails():
    ctx = await run([])
    assert ctx.failed == 'Action input is empty'
    assert ctx.completed is None


LINES = {'type': 'TEXT_LINE', 'pages': [{'pageId': 'p1', 'textLineIds': ['l1']}]}
REGIONS = {'type': 'REGION', 'pages': [{'pageId': 'p1', 'regionIds': ['r1']}]}


@pytest.mark.parametrize(
    ('parameters', 'target', 'message'),
    [
        ({'segmentation': 'words'}, None, 'Invalid parameter segmentation'),
        ({'textIndex': -1}, None, 'Invalid parameter textIndex'),
        ({'minConfidence': 2}, None, 'Invalid parameter minConfidence'),
        ({'engine': 'gone'}, None, 'Engine "gone" is not available'),
        ({'engine': 'layout-only'}, None, 'Engine "layout-only" has no text recognition'),
        ({'engine': 'recognition-only', 'segmentation': 'lines'}, None, 'has no layout detection'),
        ({'segmentation': 'keep', 'recognition': False}, None, 'Nothing to do'),
        ({'segmentation': 'regions'}, REGIONS, 'only works on whole pages'),
        ({'segmentation': 'lines'}, LINES, 'does not work on selected lines'),
        ({'recognition': False}, LINES, 'Nothing to do'),
    ],
)
async def test_impossible_runs_are_rejected(parameters, target, message):
    ctx = await run([b'ok'], parameters=parameters, target=target, xml=[EMPTY])
    assert message in ctx.failed
    assert ctx.pages == []
    assert ctx.downloads == []


async def test_engine_without_layout_can_recognize_existing_lines():
    ctx = await run([b'ok'], parameters={'engine': 'recognition-only'})
    assert ctx.statuses == ['SUCCEEDED']


async def test_waiting_runs_send_heartbeats(monkeypatch):
    monkeypatch.setattr(runner, 'QUEUE_HEARTBEAT_SECONDS', 0.05)
    await env.RUN_SEMAPHORE.acquire()
    ctx = make_context([b'ok'])
    task = asyncio.create_task(runner.runner(ctx))
    await asyncio.sleep(0.3)
    assert ctx.messages.count('Waiting for a free processor slot') >= 2
    env.RUN_SEMAPHORE.release()
    await task
    assert ctx.statuses == ['SUCCEEDED']
    assert env.RUN_SEMAPHORE._value == 1


async def test_waiting_runs_can_be_cancelled(monkeypatch):
    monkeypatch.setattr(runner, 'QUEUE_HEARTBEAT_SECONDS', 0.05)
    await env.RUN_SEMAPHORE.acquire()
    ctx = make_context([b'ok'])
    ctx.cancel_requested = True
    with pytest.raises(ActionCancelled):
        await runner.runner(ctx)
    env.RUN_SEMAPHORE.release()
    assert env.RUN_SEMAPHORE._value == 1


async def test_shutdown_reports_active_runs(fake_pool):
    ctx = make_context([b'ok', b'hang'])
    task = asyncio.create_task(runner.runner(ctx))
    while not ctx.pages:
        await asyncio.sleep(0.05)
    task.cancel()
    await runner.shutdown()
    assert ctx.failed == 'The processor was stopped'
    assert task.cancelled()
    assert runner.active_runs == set()
    assert fake_pool._idle == []


def test_settings_accept_empty_values(monkeypatch):
    for name in ('PERO_CPU_THREADS', 'PERO_BATCH_PIXELS', 'PERO_DEFAULT_ENGINE', 'ACTION_PROCESSING_TIMEOUT'):
        monkeypatch.setenv(name, '')
    settings = Settings()
    assert settings.PERO_CPU_THREADS is None
    assert settings.PERO_DEFAULT_ENGINE is None
    assert settings.ACTION_PROCESSING_TIMEOUT == 600


def test_cpu_threads_are_limited_by_default(monkeypatch):
    monkeypatch.setattr(env, 'PERO_CPU_THREADS', None)
    assert 1 <= worker.cpu_threads() <= worker.DEFAULT_MAX_CPU_THREADS
    monkeypatch.setattr(env, 'PERO_CPU_THREADS', 12)
    assert worker.cpu_threads() == 12
