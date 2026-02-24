"""
logger.py — Structured logging for the pipeline.

SAFETY DESIGN:
    - Logs go to stdout ONLY (no file handler that gets committed).
    - The old code wrote to pipeline.log which was committed as a
      git artifact. This is wasteful and pollutes the repo.
    - CI should capture stdout via GitHub Actions artifact upload.
    - Structured format for machine-parseable output.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger for the pipeline.

    SAFETY: We intentionally DO NOT create a FileHandler here.
    The old pipeline.log was committed to git, which:
    1. Bloats the repository.
    2. Creates merge conflicts on every run.
    3. Exposes internal diagnostics publicly.

    Instead, CI captures stdout/stderr via GitHub Actions artifacts.
    """
    root = logging.getLogger()

    # Prevent duplicate handlers on re-init
    if root.handlers:
        return

    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class Logger:
    """
    Thin wrapper for named loggers. Backward-compatible with old code
    that used Logger(name="PipelineRunner").
    """

    def __init__(self, name: str = "LLMIndex"):
        setup_logging()
        self.logger = logging.getLogger(name)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)
