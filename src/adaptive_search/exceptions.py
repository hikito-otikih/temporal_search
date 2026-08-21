"""Exception types raised across the adaptive-search vertical slice."""

from __future__ import annotations


class AdaptiveInputError(ValueError):
    pass


class SessionNotFoundError(KeyError):
    pass


class RevisionConflictError(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"stale session revision: expected {expected}, current revision is {actual}"
        )


class UpstreamSearchError(RuntimeError):
    """Raised when the external sparse-search service cannot be trusted."""
