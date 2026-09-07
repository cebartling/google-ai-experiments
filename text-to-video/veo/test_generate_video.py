#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
#     "tenacity",
#     "python-dotenv",
#     "pytest",
# ]
# ///
"""Tests for generate_video.py.

Run with:
    uv run test_generate_video.py
"""

import os
import sys

import pytest

import generate_video


def test_load_env_file_reads_variables_into_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env.local"
    env_file.write_text("GEMINI_API_KEY=test-key-123\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    generate_video.load_env_file(env_file)

    assert os.environ["GEMINI_API_KEY"] == "test-key-123"


def test_load_env_file_raises_when_file_is_missing(tmp_path):
    missing = tmp_path / ".env.local"

    with pytest.raises(FileNotFoundError) as excinfo:
        generate_video.load_env_file(missing)

    assert str(missing) in str(excinfo.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
