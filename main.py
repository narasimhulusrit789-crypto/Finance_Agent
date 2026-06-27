#!/usr/bin/env python
"""
Application entry point.
Run with: python main.py  OR  uvicorn main:app --reload
"""

import io
import os
import sys

import structlog
import uvicorn

# Force UTF-8 stdout/stderr on Windows to prevent UnicodeEncodeError with emoji
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from api.main import app  # noqa: F401 — re-exported for uvicorn

logger = structlog.get_logger(__name__)


def configure_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            20 if os.getenv("DEBUG", "false").lower() != "true" else 10
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


if __name__ == "__main__":
    configure_logging()

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(
        "server.starting",
        host=host,
        port=port,
        debug=debug,
        url=f"http://{host}:{port}",
    )

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info",
        access_log=True,
    )
