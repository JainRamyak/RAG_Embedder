"""
logger.py — call setup_logging() once at your entry point.

In any other file just do:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("message")
"""
import logging
import sys
from config import settings


def setup_logging() -> None:
    level = getattr(logging, settings.log_level, logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Stop noisy libraries from polluting your logs
    for noisy in ["urllib3", "httpx", "sentence_transformers", "filelock"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)