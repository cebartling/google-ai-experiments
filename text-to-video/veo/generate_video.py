#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
#     "tenacity",
# ]
# ///
"""
Generate videos with Veo via the Gemini API SDK.

Setup:
    export GEMINI_API_KEY="your-api-key"   # from Google AI Studio

Usage (uv resolves and installs deps automatically, no venv needed):
    uv run generate_video.py "A drone flies through a canyon at sunset" \
        --model veo-3.1-generate-preview \
        --aspect-ratio 16:9 \
        --resolution 1080p \
        --output my_video.mp4

    # or, if it's executable (chmod +x generate_video.py):
    ./generate_video.py "A drone flies through a canyon at sunset"
"""

import argparse
import sys
import time

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class VideoGenerationError(Exception):
    """Raised when Veo returns a terminal failure state."""


def build_client() -> genai.Client:
    # Reads GEMINI_API_KEY from the environment automatically.
    # Pass api_key="..." explicitly if you'd rather not use an env var.
    return genai.Client()


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def start_generation(client: genai.Client, prompt: str, model: str,
                      aspect_ratio: str, resolution: str):
    """Kick off the long-running video generation operation, with retries
    for transient errors (rate limits, temporary server issues)."""
    config = types.GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        resolution=resolution,
    )
    return client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=config,
    )


def poll_until_done(client: genai.Client, operation, poll_seconds: int = 10,
                     timeout_seconds: int = 600):
    """Poll the operation until it completes, fails, or times out."""
    elapsed = 0
    while not operation.done:
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"Video generation did not finish within {timeout_seconds}s"
            )
        print(f"  ...still generating ({elapsed}s elapsed)", file=sys.stderr)
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        operation = client.operations.get(operation)

    if operation.error:
        raise VideoGenerationError(f"Generation failed: {operation.error}")

    return operation


def save_videos(client: genai.Client, operation, output_path: str):
    """Download and save each generated video. If multiple videos come back,
    numbers are appended to the filename."""
    generated = operation.response.generated_videos
    if not generated:
        raise VideoGenerationError("No videos were returned in the response.")

    saved_paths = []
    for i, generated_video in enumerate(generated):
        client.files.download(file=generated_video.video)
        if len(generated) == 1:
            path = output_path
        else:
            stem, _, ext = output_path.rpartition(".")
            path = f"{stem}_{i}.{ext}" if stem else f"{output_path}_{i}"
        generated_video.video.save(path)
        saved_paths.append(path)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate a video with Veo.")
    parser.add_argument("prompt", help="Text prompt describing the video")
    parser.add_argument("--model", default="veo-3.1-generate-preview",
                         help="Veo model ID (default: veo-3.1-generate-preview)")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"],
                         help="Landscape (16:9) or portrait (9:16)")
    parser.add_argument("--resolution", default="1080p",
                         choices=["720p", "1080p", "4k"],
                         help="Output resolution")
    parser.add_argument("--output", default="output.mp4",
                         help="File path to save the generated video to")
    parser.add_argument("--poll-seconds", type=int, default=10,
                         help="Seconds between status checks")
    parser.add_argument("--timeout-seconds", type=int, default=600,
                         help="Max seconds to wait before giving up")
    args = parser.parse_args()

    client = build_client()

    print(f"Starting generation with {args.model}...", file=sys.stderr)
    try:
        operation = start_generation(
            client, args.prompt, args.model, args.aspect_ratio, args.resolution
        )
    except Exception as e:
        print(f"Failed to start generation: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        operation = poll_until_done(
            client, operation,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except (TimeoutError, VideoGenerationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        saved_paths = save_videos(client, operation, args.output)
    except VideoGenerationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
