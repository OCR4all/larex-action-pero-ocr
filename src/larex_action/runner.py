# SPDX-License-Identifier: MIT
import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

from larex_actions import ActionCancelled, ActionContext, PageResultRejected
from larex_actions.models import ActionInput, ActionPage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import env
from .engines import EngineError, engine_stages, find_engine
from .jobs import PageJob, PageOptions, Segmentation, Stage
from .pool import WorkerCrashed, WorkerError, WorkerHandle, WorkerPool, check_cancelled
from .worker import WorkerResponse, WorkerSettings, WorkerStatus

logger = logging.getLogger(__name__)

QUEUE_HEARTBEAT_SECONDS = 30
MAX_CONSECUTIVE_CRASHES = 3
SHUTDOWN_REPORT_SECONDS = 5

pool = WorkerPool()
active_runs: set[asyncio.Task] = set()


class RunParameters(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    engine: str = ''
    segmentation: Segmentation = 'auto'
    recognition: bool = True
    text_index: int = Field(default=1, alias='textIndex', ge=0)
    min_confidence: float = Field(default=0.0, alias='minConfidence', ge=0, le=1)

    @property
    def options(self) -> PageOptions:
        return PageOptions(self.segmentation, self.recognition, self.text_index, self.min_confidence)


class RunRejected(Exception):
    pass


@dataclass
class PageOutcome:
    status: str
    message: str
    xml: bytes | None = None
    warnings: list[str] = field(default_factory=list)


def read_parameters(action_input: ActionInput) -> RunParameters:
    try:
        parameters = RunParameters.model_validate(action_input.parameters)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise RunRejected(f'Invalid parameter {".".join(map(str, error["loc"]))}: {error["msg"]}') from None
    target = target_of(action_input)
    if parameters.segmentation == 'keep' and not parameters.recognition:
        raise RunRejected('Nothing to do: segmentation is "keep" and recognition is off')
    if target == 'REGION' and parameters.segmentation == 'regions':
        raise RunRejected('Detecting regions only works on whole pages; use "lines", "auto" or "keep"')
    if target == 'TEXT_LINE' and parameters.segmentation not in ('auto', 'keep'):
        raise RunRejected('Segmentation does not work on selected lines; use "auto" or "keep"')
    if target == 'TEXT_LINE' and not parameters.recognition:
        raise RunRejected('Nothing to do: selected lines can only be recognized')
    return parameters


def target_of(action_input: ActionInput) -> str:
    return action_input.target_selection.type if action_input.target_selection else 'PAGE'


def worker_settings(parameters: RunParameters, target: str) -> WorkerSettings:
    try:
        engine = find_engine(parameters.engine)
        available = engine_stages(engine)
    except EngineError as exc:
        raise RunRejected(str(exc)) from None
    wanted: set[Stage] = set()
    if parameters.segmentation != 'keep' and target != 'TEXT_LINE':
        wanted.add('layout')
    if parameters.recognition:
        wanted.add('recognition')
    if parameters.recognition and 'recognition' not in available:
        raise RunRejected(f'Engine "{engine.name}" has no text recognition')
    if parameters.segmentation in ('regions', 'lines') and 'layout' not in available:
        raise RunRejected(f'Engine "{engine.name}" has no layout detection')
    return WorkerSettings(engine, engine.fingerprint, frozenset(wanted))


async def load_job(
    ctx: ActionContext, action_input: ActionInput, page: ActionPage, options: PageOptions, first: bool
) -> PageJob | str:
    if not page.images:
        return 'Page has no image'
    target = target_of(action_input)
    selection = None
    if action_input.target_selection is not None:
        selection = next((item for item in action_input.target_selection.pages if item.page_id == page.id), None)
    if target != 'PAGE' and not page.xml:
        return 'Page has no PAGE XML'
    image = await ctx.download_bytes(page.images[0])
    xml = await ctx.download_bytes(page.xml[0]) if page.xml else None
    return PageJob(
        image=image,
        xml=xml,
        image_name=page.images[0].file_name,
        options=options,
        target=target,
        region_ids=tuple(selection.region_ids) if selection else (),
        line_ids=tuple(selection.text_line_ids) if selection else (),
        first_of_run=first,
    )


class PageLoader:
    def __init__(self, ctx: ActionContext, action_input: ActionInput, options: PageOptions) -> None:
        self._ctx = ctx
        self._input = action_input
        self._options = options
        self._tasks: dict[int, asyncio.Task[PageJob | str]] = {}

    async def get(self, index: int) -> PageJob | str:
        last = min(index + env.PERO_PREFETCH_PAGES, len(self._input.pages) - 1)
        for position in range(index, last + 1):
            if position not in self._tasks:
                page = self._input.pages[position]
                self._tasks[position] = asyncio.create_task(
                    load_job(self._ctx, self._input, page, self._options, position == 0)
                )
        return await self._tasks.pop(index)

    async def close(self) -> None:
        await cancel(*self._tasks.values())
        self._tasks.clear()


class ResultUploader:
    def __init__(self, ctx: ActionContext) -> None:
        self._ctx = ctx
        self._task: asyncio.Task[bool] | None = None
        self.succeeded = 0

    async def submit(self, page: ActionPage, outcome: PageOutcome) -> None:
        await self.flush()
        self._task = asyncio.create_task(self._upload(page, outcome))

    async def flush(self) -> None:
        task, self._task = self._task, None
        if task is not None and await task:
            self.succeeded += 1

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            await cancel(task)

    async def _upload(self, page: ActionPage, outcome: PageOutcome) -> bool:
        results = None
        if outcome.xml is not None:
            results = self._ctx.result_builder()
            results.add_xml_bytes(page_id=page.id, content=outcome.xml, file_name=f'{page.name or "page"}.xml')
        try:
            await self._ctx.submit_page_results(
                page.id,
                results,
                outcome.message,
                page_status=outcome.status,
                reason=outcome.message,
                warnings=outcome.warnings or None,
            )
        except PageResultRejected as exc:
            logger.warning(f'LAREX rejected the result of {page.name}: {exc}')
            return False
        return outcome.status == 'SUCCEEDED'


async def cancel(*tasks: asyncio.Task) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def run_slot(ctx: ActionContext) -> AsyncIterator[None]:
    semaphore = env.RUN_SEMAPHORE
    acquiring = asyncio.create_task(semaphore.acquire())
    try:
        while not (await asyncio.wait({acquiring}, timeout=QUEUE_HEARTBEAT_SECONDS))[0]:
            await ctx.heartbeat(status_message='Waiting for a free processor slot', raise_on_cancel=True)
    except BaseException:
        if not acquiring.cancel() and not acquiring.cancelled():
            semaphore.release()
        with suppress(asyncio.CancelledError):
            await acquiring
        raise
    try:
        yield
    finally:
        semaphore.release()


async def process_page(ctx: ActionContext, handle: WorkerHandle, job: PageJob, label: str) -> tuple[PageOutcome, bool]:
    await handle.send(job)
    result = await handle.receive(ctx, env.ACTION_PROCESSING_TIMEOUT)
    if isinstance(result, WorkerStatus):
        logger.error(f'{label} failed: {result.message}')
        return PageOutcome('FAILED', f'{label} failed: {result.message}'), result.restart
    if not isinstance(result, WorkerResponse):
        raise WorkerError(f'Worker returned an unknown result type: {type(result).__name__}')
    if result.skipped is not None:
        logger.info(f'{label} skipped: {result.skipped}')
        return PageOutcome('SKIPPED', f'{label} skipped: {result.skipped}', warnings=result.warnings), False
    logger.debug(f'{label} finished with {result.lines} lines')
    return PageOutcome('SUCCEEDED', f'{label}: {result.lines} line(s)', result.xml, result.warnings), False


async def _process_pages(
    ctx: ActionContext, action_input: ActionInput, settings: WorkerSettings, options: PageOptions
) -> int:
    total = len(action_input.pages)
    loader = PageLoader(ctx, action_input, options)
    uploader = ResultUploader(ctx)
    handle: WorkerHandle | None = None
    crashes = 0
    healthy = False
    try:
        for index, page in enumerate(action_input.pages):
            label = f'[{index + 1}/{total}] {page.name}'
            await check_cancelled(ctx)
            job = await loader.get(index)
            if handle is None:
                handle = await pool.acquire(ctx, settings)
            await ctx.heartbeat(
                progress_percent=index * 100 // total,
                status_message=f'{page.name} ({index + 1}/{total})',
                raise_on_cancel=True,
            )
            if isinstance(job, str):
                await uploader.submit(page, PageOutcome('FAILED', f'{label} failed: {job}'))
                continue
            restart = False
            crashed = False
            try:
                outcome, restart = await process_page(ctx, handle, job, label)
            except TimeoutError:
                message = f'{label} timed out after {env.ACTION_PROCESSING_TIMEOUT} seconds'
                logger.error(message)
                if env.ACTION_TIMEOUT_MODE == 'fail_action':
                    raise WorkerError(message) from None
                outcome, restart = PageOutcome('FAILED', message), True
            except WorkerCrashed as exc:
                logger.error(f'{label} failed: {exc}')
                outcome, restart, crashed = PageOutcome('FAILED', f'{label} failed: {exc}'), True, True
            crashes = crashes + 1 if crashed else 0
            if restart:
                await handle.kill()
                handle = None
            await uploader.submit(page, outcome)
            if crashes >= MAX_CONSECUTIVE_CRASHES:
                await uploader.flush()
                raise WorkerError(f'Worker crashed on {crashes} pages in a row: {outcome.message}')
        await uploader.flush()
        healthy = True
        return uploader.succeeded
    finally:
        await loader.close()
        await uploader.close()
        if handle is not None:
            if healthy:
                await pool.release(handle)
            else:
                await handle.kill()


async def _run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.fail('Action input is empty')
        return
    try:
        parameters = read_parameters(action_input)
        settings = worker_settings(parameters, target_of(action_input))
    except RunRejected as exc:
        await ctx.fail(str(exc))
        return

    total = len(action_input.pages)
    logger.info(
        f'Starting run for {total} page(s) with {settings.engine.name}: '
        f'segmentation {parameters.segmentation}, recognition {parameters.recognition}'
    )
    try:
        succeeded = await _process_pages(ctx, action_input, settings, parameters.options)
    except ActionCancelled:
        logger.info('Action cancelled')
        raise
    except WorkerError as exc:
        logger.error(str(exc))
        await ctx.fail(str(exc))
        return
    message = f'Processed {succeeded}/{total} page(s) with {settings.engine.name}'
    logger.info(message)
    await ctx.complete(message=message)


async def runner(ctx: ActionContext) -> None:
    task = asyncio.current_task()
    active_runs.add(task)
    try:
        async with run_slot(ctx):
            await _run(ctx)
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.wait_for(ctx.fail('The processor was stopped'), SHUTDOWN_REPORT_SECONDS)
        raise
    finally:
        active_runs.discard(task)


async def shutdown() -> None:
    if active_runs:
        await asyncio.wait(active_runs, timeout=2 * SHUTDOWN_REPORT_SECONDS)
    await pool.close()
