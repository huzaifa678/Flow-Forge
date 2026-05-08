import logging

def setup_logger() -> logging.Logger:
    """
    Configure and return application logger.
    """

    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)


    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
