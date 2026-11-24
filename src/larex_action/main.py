# SPDX-License-Identifier: MIT
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from larex_actions.fastapi import create_larex_action_app

from .config import env
from .engines import engine_choices
from .logs import configure_logging
from .runner import runner, shutdown


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    await shutdown()


configure_logging()

app = create_larex_action_app(
    app=FastAPI(title=f'LAREX Action Processor: {env.LAREX_PROCESSOR_ID}', lifespan=lifespan),
    handler=runner,
    processor_id=env.LAREX_PROCESSOR_ID,
    dispatch_secret=env.LAREX_DISPATCH_HMAC_SECRET,
    route_prefixes=env.LAREX_ROUTE_PREFIXES,
    allowed_callback_origins=env.LAREX_ALLOWED_CALLBACK_ORIGINS,
    parameter_value_providers={'engines': engine_choices},
)
