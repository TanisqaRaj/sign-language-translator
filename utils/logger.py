# ─────────────────────────────────────────────────────────────────────────────
# utils/logger.py
# Centralised logging setup used by every module in the project.
# Creates a rotating file handler so logs never grow unbounded, and also
# streams INFO+ messages to the console for easy debugging.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
from logging.handlers import RotatingFileHandler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import LOG_DIR, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured Logger instance.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Logger with both console and rotating file handlers attached.
    """

    # Create logs/ directory if it doesn't exist yet
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler (INFO and above) ──────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── Rotating File Handler (DEBUG and above, max 5 MB × 3 backups) ─────────
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
