"""Process-wide bootstrap: load .env before any module reads its variables."""

from dotenv import load_dotenv

load_dotenv()
