import logging
import sys


def setup_logger(
    name: str = "Test",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    logger = logging.Logger(name=name, level=level)
    logger.propagate = False

    if logger.handlers:
        for h in logger.handlers:
            h.setLevel(level)
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)

    return logger
