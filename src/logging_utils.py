"""Shared logging setup: every pipeline script writes a live, tail-able log file
in addition to stdout, so `tail -f logs/<...>.log` shows progress for
background/long-running runs without relying on shell redirection.
"""
import logging
from pathlib import Path


def setup_logging(log_file, name=None):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # FileHandler.emit() (via StreamHandler.emit) flushes after every record,
    # so the file is safe to `tail -f` live rather than only readable once
    # buffers are flushed at process exit.
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.info(f"Logging to {log_file}")
    return logger


def default_log_path(ref_path, logs_dir="logs"):
    return str(Path(logs_dir) / f"{Path(ref_path).stem}.log")
