# SPDX-License-Identifier: MIT
import asyncio

import httpx
from larex_actions import ActionCancelled, PageResultRejected, ResultBuilder
from larex_actions.models import ActionInput


class FakeContext:
    def __init__(self, action_input, files):
        self.input = action_input
        self.files = files
        self.pages = []
        self.messages = []
        self.completed = None
        self.failed = None
        self.cancel_requested = False
        self.cancel_after_pages = None
        self.rejected_pages = set()
        self.downloads = []

    async def pull_input(self):
        return self.input

    async def check_cancelled(self):
        if self.cancel_requested:
            raise ActionCancelled('cancelled')

    async def heartbeat(self, progress_percent=None, status_message=None, *, raise_on_cancel=False):
        self.messages.append(status_message)
        if raise_on_cancel:
            await self.check_cancelled()

    async def download_bytes(self, file):
        self.downloads.append(file.id)
        await asyncio.sleep(0)
        content = self.files[file.id]
        if isinstance(content, Exception):
            raise content
        return content

    def result_builder(self):
        return ResultBuilder()

    async def submit_page_results(
        self, page_id, results=None, message=None, *, page_status=None, reason=None, warnings=None
    ):
        if page_id in self.rejected_pages:
            request = httpx.Request('POST', 'http://127.0.0.1/results')
            raise PageResultRejected('invalid PAGE XML', request=request, response=httpx.Response(422, request=request))
        self.pages.append(
            {
                'page_id': page_id,
                'status': page_status,
                'reason': reason,
                'warnings': warnings,
                'files': list(results._files) if results is not None else [],
            }
        )
        if self.cancel_after_pages == len(self.pages):
            self.cancel_requested = True

    async def complete(self, results=None, message=None, *, run_report=None):
        self.completed = message

    async def fail(self, message, *, log=None, run_report=None):
        self.failed = message

    @property
    def statuses(self):
        return [page['status'] for page in self.pages]


def file_ref(file_id):
    return {'id': file_id, 'fileName': f'{file_id}.png', 'downloadUrl': f'http://127.0.0.1/{file_id}'}


def make_context(images, *, parameters=None, target=None, xml=()):
    files = {}
    pages = []
    for number, image in enumerate(images, start=1):
        page = {'id': f'p{number}', 'name': f'Page {number}', 'images': [], 'xml': []}
        if image is not None:
            files[f'img{number}'] = image
            page['images'].append(file_ref(f'img{number}'))
        if number <= len(xml) and xml[number - 1] is not None:
            files[f'xml{number}'] = xml[number - 1]
            page['xml'].append(file_ref(f'xml{number}'))
        pages.append(page)
    payload = {
        'protocolVersion': 1,
        'runId': 'run-1',
        'processorKey': 'pero-ocr',
        'projectId': 'project-1',
        'parameters': parameters or {},
        'pages': pages,
        'capabilities': {'incrementalPageResults': True, 'customFileResults': True, 'runReports': True},
    }
    if target is not None:
        payload['targetSelection'] = target
    return FakeContext(ActionInput.model_validate(payload), files)
