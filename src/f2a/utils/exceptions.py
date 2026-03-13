"""커스텀 예외 정의."""


class F2AError(Exception):
    """f2a 라이브러리의 기본 예외."""


class UnsupportedFormatError(F2AError):
    """지원하지 않는 파일 포맷."""

    def __init__(self, source: str, detected: str | None = None) -> None:
        msg = f"지원하지 않는 파일 포맷입니다: {source}"
        if detected:
            msg += f" (감지된 형식: {detected})"
        super().__init__(msg)


class DataLoadError(F2AError):
    """데이터 로딩 실패."""

    def __init__(self, source: str, reason: str = "") -> None:
        msg = f"데이터를 로딩할 수 없습니다: {source}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg)


class EmptyDataError(F2AError):
    """빈 데이터셋."""

    def __init__(self, source: str) -> None:
        super().__init__(f"데이터셋이 비어 있습니다: {source}")
