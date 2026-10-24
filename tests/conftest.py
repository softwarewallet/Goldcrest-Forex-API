"""Shared pytest setup for deterministic, isolated database state.

The application constructs its SQLAlchemy engine when app.db is imported. Set the
database path here, before test modules import app.main, so test runs never read
or mutate a developer's persistent research/API database and cannot inherit stale
idempotency records from earlier pytest runs.
"""
import os
import tempfile
from pathlib import Path

_TEST_DATABASE_DIRECTORY = Path(tempfile.mkdtemp(prefix="goldcrest-forex-api-pytest-"))
os.environ["DATABASE_PATH"] = str(_TEST_DATABASE_DIRECTORY / "test.sqlite3")
