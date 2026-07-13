"""Logger configuration utilities for the LeafFusionNet project."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured project logger.

    Parameters
    ----------
    name : str
        Name of the logger to retrieve or configure.

    Returns
    -------
    logging.Logger
        Logger configured with an INFO level, stream handler, formatter, and
        disabled propagation to the root logger.

    Notes
    -----
    The logger is configured only when it has no existing handlers. This avoids
    duplicate log messages when the function is called repeatedly with the same
    logger name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


__all__ = ["get_logger"]
