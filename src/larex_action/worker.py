# SPDX-License-Identifier: MIT
import logging
import os
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Literal

from .config import env
from .engines import Engine
from .errors import PageInputError, PageSkipped
from .jobs import PageJob, Stage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if env.ACTION_DEBUG else logging.INFO)

MAX_MESSAGE_LENGTH = 500
DEFAULT_MAX_CPU_THREADS = 4


@dataclass(frozen=True)
class WorkerSettings:
    engine: Engine
    fingerprint: tuple[tuple[str, int, int], ...]
    stages: frozenset[Stage]


@dataclass
class WorkerResponse:
    xml: bytes | None
    warnings: list[str]
    lines: int = 0
    skipped: str | None = None


@dataclass
class WorkerStatus:
    type: Literal['loaded', 'error']
    message: str | None = None
    restart: bool = False


def cpu_threads() -> int:
    if env.PERO_CPU_THREADS:
        return env.PERO_CPU_THREADS
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return min(available, DEFAULT_MAX_CPU_THREADS)


def describe(exc: Exception) -> str:
    return f'{type(exc).__name__}: {exc}'[:MAX_MESSAGE_LENGTH]


def _report(connection: Connection, status: WorkerStatus) -> bool:
    try:
        connection.send(status)
    except (BrokenPipeError, EOFError, OSError):
        logger.warning('Could not send worker status to parent')
        return False
    return True


def worker(connection: Connection, settings: WorkerSettings) -> None:
    logger.debug('Initialize worker')

    try:
        threads = cpu_threads()
        os.environ['OMP_NUM_THREADS'] = str(threads)
        import torch

        from .pipeline import Pipeline

        pipeline = Pipeline(
            settings.engine,
            settings.stages,
            device=env.PERO_DEVICE,
            cpu_threads=threads,
            batch_pixels=env.PERO_BATCH_PIXELS,
            max_megapixels=env.PERO_MAX_IMAGE_MEGAPIXELS,
        )
        connection.send(WorkerStatus('loaded'))
    except Exception as exc:
        logger.exception('Failed to load worker')
        _report(connection, WorkerStatus('error', describe(exc)))
        connection.close()
        return

    while True:
        try:
            request = connection.recv()
        except EOFError:
            logger.debug('Parent closed pipe; shutting down worker')
            break
        if request is None:
            logger.debug('Parent sent shutdown signal; shutting down worker')
            break
        if not isinstance(request, PageJob):
            logger.error(f'Received unexpected request: {type(request).__name__}')
            _report(connection, WorkerStatus('error', f'Unexpected request: {type(request).__name__}'))
            continue

        try:
            result = pipeline.process(request)
            connection.send(WorkerResponse(result.xml, result.warnings, result.lines))
        except PageSkipped as exc:
            connection.send(WorkerResponse(None, [], skipped=str(exc)))
        except PageInputError as exc:
            if not _report(connection, WorkerStatus('error', str(exc))):
                break
        except torch.cuda.OutOfMemoryError:
            logger.exception('Out of GPU memory')
            torch.cuda.empty_cache()
            if not _report(connection, WorkerStatus('error', 'Out of GPU memory')):
                break
        except Exception as exc:
            logger.exception('Failed to process element')
            broken = isinstance(exc, RuntimeError) and 'CUDA' in str(exc)
            if not _report(connection, WorkerStatus('error', describe(exc), restart=broken)):
                break

    connection.close()
    logger.info('Worker stopped successfully')
