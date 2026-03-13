"""f2a 로깅 설정."""

import logging

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환합니다.

    Args:
        name: 로거 이름 (보통 ``__name__``).

    Returns:
        설정된 :class:`logging.Logger` 인스턴스.
    """
    logger = logging.getLogger(f"f2a.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
