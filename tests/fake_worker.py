# SPDX-License-Identifier: MIT
import os
import time

from page_fixtures import page_xml

from larex_action.worker import WorkerResponse, WorkerStatus


def worker(connection, settings):
    if settings.engine.name == 'full' and os.environ.get('FAKE_WORKER_LOADING') == 'error':
        connection.send(WorkerStatus('error', 'EngineError: broken'))
        return
    if os.environ.get('FAKE_WORKER_LOADING') == 'crash':
        os._exit(7)
    if os.environ.get('FAKE_WORKER_LOADING') == 'hang':
        time.sleep(60)
    connection.send(WorkerStatus('loaded'))
    while True:
        try:
            job = connection.recv()
        except EOFError:
            return
        if job is None:
            return
        match job.image:
            case b'crash':
                os._exit(3)
            case b'hang':
                time.sleep(60)
            case b'fail':
                connection.send(WorkerStatus('error', 'ValueError: bad page'))
            case b'restart':
                connection.send(WorkerStatus('error', 'RuntimeError: CUDA error', restart=True))
            case b'skip':
                connection.send(WorkerResponse(None, ['careful'], skipped='No text found on the page'))
            case b'unknown':
                connection.send('?')
            case _:
                body = f'<!--{os.getpid()} {job.target} {job.region_ids} {job.line_ids} {job.first_of_run}-->'
                connection.send(WorkerResponse(page_xml(body), ['warned'], 4))
