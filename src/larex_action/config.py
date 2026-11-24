# SPDX-License-Identifier: MIT
import asyncio
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True)

    LAREX_PROCESSOR_ID: str
    LAREX_DISPATCH_HMAC_SECRET: str
    LAREX_ALLOWED_CALLBACK_ORIGINS: list[str]
    LAREX_ROUTE_PREFIXES: list[str] = []
    LAREX_SDK_TRANSPORT_DEBUG: bool = False

    ACTION_DEBUG: bool = False
    ACTION_MAX_CONCURRENT_RUNS: PositiveInt = 1
    ACTION_LOADING_TIMEOUT: PositiveInt | None = 180
    ACTION_PROCESSING_TIMEOUT: PositiveInt | None = 600
    ACTION_TIMEOUT_MODE: Literal['fail_action', 'fail_page'] = 'fail_page'
    ACTION_IDLE_TIMEOUT: NonNegativeInt = 600

    PERO_MODEL_DIR: Path = Path('/models')
    PERO_DEFAULT_ENGINE: str | None = None
    PERO_DEVICE: str = Field(default='auto', pattern=r'^(auto|cpu|cuda(:\d+)?)$')
    PERO_CPU_THREADS: PositiveInt | None = None
    PERO_BATCH_PIXELS: PositiveInt | None = None
    PERO_MAX_IMAGE_MEGAPIXELS: PositiveInt = 200
    PERO_PREFETCH_PAGES: NonNegativeInt = 2

    @cached_property
    def RUN_SEMAPHORE(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(self.ACTION_MAX_CONCURRENT_RUNS)


env = Settings()
