"""Logging utilities to keep experiment outputs consistent."""

import logging
from typing import Optional


def get_logger(name: str = "cis_factual_llm", level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger for experiments.

    The logger should write to stdout and optionally to a file, using a concise
    format suitable for iterative research runs.
    """
    raise NotImplementedError
