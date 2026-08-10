"""Exception types raised by the query-rewrite service."""

from __future__ import annotations


class ConfigurationError(RuntimeError):
    '''Raised when the server is missing required Ollama configuration.'''


class OllamaServiceError(RuntimeError):
    '''Raised when Ollama cannot return a usable structured analysis.'''


class OllamaTimeoutError(OllamaServiceError):
    '''Raised when an Ollama request exceeds the configured timeout.'''


class OllamaRateLimitError(OllamaServiceError):
    '''Raised when Ollama rejects a request because of rate limiting.'''


class SemanticValidationError(ValueError):
    '''Raised when valid structured output lacks standalone semantic context.'''
