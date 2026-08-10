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


class LiveRefinementError(RuntimeError):
    """Base error for live refinement failures."""


class LiveRefinementUnavailableError(LiveRefinementError):
    """Decoder or model runtime is unavailable before any work starts."""


class LiveRefinementConfigurationError(LiveRefinementError):
    """Injected model identity differs from the pinned session spec."""


class LiveRefinementDataError(LiveRefinementError):
    """A provider response violates the canonical frame contract."""
