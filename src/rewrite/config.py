"""Ollama connection defaults and .env bootstrap for the rewrite service."""

from __future__ import annotations

from dotenv import load_dotenv

OLLAMA_API_URL = 'https://ollama.com/api/chat'
DEFAULT_OLLAMA_MODEL = 'gpt-oss:20b'
OLLAMA_TIMEOUT_SECONDS = 180.0


def load_ollama_env() -> None:
    '''Load .env from the process's working directory upward (idempotent).'''
    load_dotenv()
