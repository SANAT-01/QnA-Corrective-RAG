"""Structured logging setup, configured once and imported everywhere else.

Call configure_logging() once at process startup (done at the bottom of
this module, so just importing qna.logging_config is enough), then get a
logger per module with logging.getLogger(__name__) as usual.
"""

import logging

from qna.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


configure_logging()
