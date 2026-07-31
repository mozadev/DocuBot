"""
Application logging.

Human-readable and coloured on a TTY for local development, plain in the log
file. A production deployment would swap the console formatter for JSON so the
lines are queryable in CloudWatch or Cloud Logging; that is a formatter change
and nothing else, which is why formatting is isolated here.
"""

from __future__ import annotations

import logging
import sys
from functools import wraps

from config.settings import settings

CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d - %(message)s"

_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Colours the level name when stdout is a terminal."""

    def __init__(self, fmt: str) -> None:
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self._use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            # Copy the record: mutating levelname in place corrupts the file
            # handler's output, since both handlers format the same object.
            record = logging.makeLogRecord(record.__dict__)
            color = _COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def setup_logger(name: str = "docubot", level: str | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(getattr(logging, (level or settings.log_level).upper(), logging.INFO))

    if log.handlers:
        return log

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColorFormatter(CONSOLE_FORMAT))
    log.addHandler(console)

    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        log.addHandler(file_handler)

    log.propagate = False
    return log


logger = setup_logger()


def log_function_call(func):
    """Debug-level entry/exit logging. Errors are logged and re-raised."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("-> %s", func.__name__)
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error("%s failed: %s", func.__name__, e)
            raise
        logger.debug("<- %s", func.__name__)
        return result

    return wrapper
