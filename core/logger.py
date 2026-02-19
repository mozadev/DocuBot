"""
Sistema de logging para DocuBot AI.
"""

import logging
import sys
from datetime import datetime
from typing import Optional
from config.settings import settings


class CustomFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m',
    }

    def format(self, record):
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(name: str = "docubot", level: Optional[str] = None) -> logging.Logger:
    _logger = logging.getLogger(name)
    _logger.setLevel(getattr(logging, level or settings.log_level))

    if _logger.handlers:
        return _logger

    console_fmt = CustomFormatter('%(timestamp)s - %(name)s - %(levelname)s - %(message)s')
    file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_fmt)
    _logger.addHandler(ch)

    if settings.log_file:
        fh = logging.FileHandler(settings.log_file)
        fh.setFormatter(file_fmt)
        _logger.addHandler(fh)

    return _logger


logger = setup_logger()


def log_function_call(func):
    """Decorador para logging automático de llamadas a funciones."""
    def wrapper(*args, **kwargs):
        logger.debug(f"Llamando: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completado: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error en {func.__name__}: {e}")
            raise
    return wrapper
