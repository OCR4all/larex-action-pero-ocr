# SPDX-License-Identifier: MIT
import logging

from .config import env


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logging.getLogger('larex_actions.transport').setLevel(
        level=logging.DEBUG if env.LAREX_SDK_TRANSPORT_DEBUG else logging.INFO
    )
    logging.getLogger('larex_action').setLevel(level=logging.DEBUG if env.ACTION_DEBUG else logging.INFO)
