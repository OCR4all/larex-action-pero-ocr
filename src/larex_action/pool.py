# SPDX-License-Identifier: MIT
import asyncio
import logging
import multiprocessing
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnProcess

from larex_actions import ActionCancelled, ActionContext

from .config import env
from .worker import WorkerSettings, WorkerStatus, worker

logger = logging.getLogger(__name__)

POLL_SECONDS = 0.25
STOP_SECONDS = 10


class WorkerError(Exception):
    pass


class WorkerCrashed(WorkerError):
    pass


async def check_cancelled(ctx: ActionContext) -> None:
    try:
        await ctx.check_cancelled()
    except ActionCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning(f'ctx.check_cancelled() raised an error: {exc}')


@dataclass(eq=False)
class WorkerHandle:
    settings: WorkerSettings
    process: SpawnProcess
    connection: Connection
    expiry: asyncio.TimerHandle | None = field(default=None, repr=False)

    @classmethod
    def start(cls, settings: WorkerSettings) -> 'WorkerHandle':
        context = multiprocessing.get_context('spawn')
        connection, child_connection = context.Pipe()
        process = context.Process(target=worker, args=(child_connection, settings), daemon=True)
        process.start()
        child_connection.close()
        return cls(settings, process, connection)

    @property
    def alive(self) -> bool:
        return self.process.is_alive()

    def serves(self, settings: WorkerSettings) -> bool:
        return (
            self.settings.engine == settings.engine
            and self.settings.fingerprint == settings.fingerprint
            and self.settings.stages >= settings.stages
        )

    async def send(self, message: object) -> None:
        try:
            await asyncio.to_thread(self.connection.send, message)
        except (BrokenPipeError, EOFError, OSError):
            raise await self._crashed() from None

    async def receive(self, ctx: ActionContext, timeout: float | None) -> object:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout if timeout is not None else None
        while True:
            await check_cancelled(ctx)
            wait = POLL_SECONDS
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(f'Worker timed out after {timeout} seconds')
                wait = min(wait, remaining)
            try:
                if await asyncio.to_thread(self.connection.poll, wait):
                    return await asyncio.to_thread(self.connection.recv)
            except (EOFError, OSError):
                raise await self._crashed() from None
            if not self.alive and not self.connection.poll(0):
                raise await self._crashed()

    async def wait_until_loaded(self, ctx: ActionContext) -> None:
        try:
            status = await self.receive(ctx, env.ACTION_LOADING_TIMEOUT)
        except TimeoutError:
            raise WorkerError(f'Worker did not load within {env.ACTION_LOADING_TIMEOUT} seconds') from None
        except WorkerCrashed as exc:
            raise WorkerError(f'{exc} while loading') from None
        if isinstance(status, WorkerStatus) and status.type == 'loaded':
            logger.info('Worker loaded')
            return
        if isinstance(status, WorkerStatus):
            raise WorkerError(f'Worker failed to load: {status.message}')
        raise WorkerError(f'Worker sent an unexpected message while loading: {type(status).__name__}')

    async def stop(self) -> None:
        with suppress(BrokenPipeError, EOFError, OSError):
            self.connection.send(None)
        await asyncio.to_thread(self.process.join, STOP_SECONDS)
        await self.kill()

    async def kill(self) -> None:
        if self.alive:
            self.process.terminate()
            await asyncio.to_thread(self.process.join, STOP_SECONDS)
        if self.alive:
            logger.warning('Worker ignored SIGTERM; killing process')
            self.process.kill()
        await asyncio.to_thread(self.process.join)
        self.connection.close()

    async def _crashed(self) -> WorkerCrashed:
        await asyncio.to_thread(self.process.join, 1)
        return WorkerCrashed(f'Worker exited unexpectedly with code {self.process.exitcode}')


class WorkerPool:
    def __init__(self) -> None:
        self._idle: list[WorkerHandle] = []
        self._closing: set[asyncio.Task[None]] = set()

    async def acquire(self, ctx: ActionContext, settings: WorkerSettings) -> WorkerHandle:
        handle = next((item for item in self._idle if item.alive and item.serves(settings)), None)
        if handle is not None:
            self._take(handle)
            logger.debug('Reusing loaded worker')
            return handle
        for stale in [self._take(item) for item in list(self._idle)]:
            await stale.stop()
        await ctx.heartbeat(status_message=f'Loading {settings.engine.name}', raise_on_cancel=True)
        handle = WorkerHandle.start(settings)
        try:
            await handle.wait_until_loaded(ctx)
        except BaseException:
            await handle.kill()
            raise
        return handle

    async def release(self, handle: WorkerHandle) -> None:
        if env.ACTION_IDLE_TIMEOUT == 0 or not handle.alive:
            await handle.stop()
            return
        handle.expiry = asyncio.get_running_loop().call_later(env.ACTION_IDLE_TIMEOUT, self._expire, handle)
        self._idle.append(handle)

    async def close(self) -> None:
        for handle in [self._take(item) for item in list(self._idle)]:
            await handle.kill()
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)

    def _take(self, handle: WorkerHandle) -> WorkerHandle:
        self._idle.remove(handle)
        if handle.expiry is not None:
            handle.expiry.cancel()
            handle.expiry = None
        return handle

    def _expire(self, handle: WorkerHandle) -> None:
        if handle in self._idle:
            logger.info('Stopping idle worker')
            self._take(handle)
            self._background(handle.stop)

    def _background(self, action: Callable[[], Awaitable[None]]) -> None:
        task = asyncio.ensure_future(action())
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
