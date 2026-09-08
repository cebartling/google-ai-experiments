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
from pathlib import Path

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


def test_resolve_prompt_returns_inline_prompt():
    assert generate_video.resolve_prompt("a drone over a canyon", None) == "a drone over a canyon"


def test_resolve_prompt_reads_file_and_preserves_internal_newlines(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a drone over a canyon,\ngolden hour, slow push in\n")

    assert generate_video.resolve_prompt(None, prompt_file) == (
        "a drone over a canyon,\ngolden hour, slow push in"
    )


def test_resolve_prompt_rejects_both_sources():
    with pytest.raises(ValueError, match="not both"):
        generate_video.resolve_prompt("inline", Path("prompt.txt"))


def test_resolve_prompt_requires_one_source():
    with pytest.raises(ValueError, match="[Pp]rovide"):
        generate_video.resolve_prompt(None, None)


def test_resolve_prompt_raises_when_file_is_missing(tmp_path):
    missing = tmp_path / "nope.txt"

    with pytest.raises(FileNotFoundError) as excinfo:
        generate_video.resolve_prompt(None, missing)

    assert str(missing) in str(excinfo.value)


def test_resolve_prompt_rejects_blank_file(tmp_path):
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\n")

    with pytest.raises(ValueError, match="empty"):
        generate_video.resolve_prompt(None, blank)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
