"""Custom exception definitions."""


class F2AError(Exception):
    """Base exception for the f2a library."""


class UnsupportedFormatError(F2AError):
    """Unsupported file format."""

    def __init__(self, source: str, detected: str | None = None) -> None:
        msg = f"Unsupported file format: {source}"
        if detected:
            msg += f" (detected: {detected})"
        super().__init__(msg)


class DataLoadError(F2AError):
    """Data loading failure."""

    def __init__(self, source: str, reason: str = "") -> None:
        msg = f"Failed to load data: {source}"
        if reason:
            msg += f" — {reason}"
        super().__init__(msg)


class EmptyDataError(F2AError):
    """Empty dataset."""

    def __init__(self, source: str) -> None:
        super().__init__(f"Dataset is empty: {source}")
