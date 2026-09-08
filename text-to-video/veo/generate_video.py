#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai",
#     "tenacity",
#     "python-dotenv",
# ]
# ///
"""
Generate videos with Veo via the Gemini API SDK.

Setup:
    Copy the template next to this script and fill in your Google AI
    Studio key:

        cp .env.example .env.local

Usage (uv resolves and installs deps automatically, no venv needed):
    uv run generate_video.py "A drone flies through a canyon at sunset" \
        --model veo-3.1-generate-preview \
        --aspect-ratio 16:9 \
        --resolution 1080p \
        --output my_video.mp4

    # or, if it's executable (chmod +x generate_video.py):
    ./generate_video.py "A drone flies through a canyon at sunset"

    # longer prompts are easier to keep in a file:
    uv run generate_video.py --prompt-file prompt.txt

    # check what would be sent without spending a generation:
    ./generate_video.py "a drone over a canyon" --dry-run

    # animate a still, or interpolate between two stills:
    ./generate_video.py "the fog rolls in" --image start.png --last-frame end.png

    # a looping clip: ends on the frame it started from
    ./generate_video.py "a candle flame flickers" --image frame.png --loop \
        --duration 8 --resolution 720p

    # ...and crossfade the tail over the head to hide the remaining seam
    # (needs ffmpeg; the clip comes out 0.5s shorter):
    ./generate_video.py "a candle flame flickers" --image frame.png --loop \
        --crossfade --duration 8 --resolution 720p

See README.md for the full flag list and the constraints the API enforces.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


class VideoGenerationError(Exception):
    """Raised when Veo returns a terminal failure state."""


ENV_FILE = Path(__file__).resolve().parent / ".env.local"

# The API accepts JPEG and PNG only.
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

REFERENCE_TYPES = ("asset", "style")

MAX_REFERENCE_IMAGES = 3

# Veo has no loop parameter. A seamless loop comes from ending on the frame you
# started with, so the drift that would give the seam away has to be suppressed
# through the negative prompt instead.
LOOP_NEGATIVE_PROMPT = (
    "camera movement, camera pan, zoom, dolly, handheld shake, cuts, "
    "scene change, fade to black, fade in, text overlay"
)


def load_env_file(env_path: Path) -> None:
    """Load environment variables from env_path.

    Raises FileNotFoundError if the file is missing, so a misconfigured
    checkout fails immediately instead of at the first API call.
    """
    if not env_path.is_file():
        raise FileNotFoundError(
            f"Environment file not found: {env_path}. "
            "Create it with a line reading GEMINI_API_KEY=your-api-key."
        )
    load_dotenv(env_path, override=False)


def resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    """Return the prompt text from either the inline argument or a file.

    Exactly one source must be supplied. File content keeps its internal
    line breaks; only surrounding whitespace is trimmed.
    """
    if prompt is not None and prompt_file is not None:
        raise ValueError("Provide a prompt or --prompt-file, not both.")
    if prompt is None and prompt_file is None:
        raise ValueError("Provide a prompt argument or --prompt-file.")

    if prompt is not None:
        return prompt

    if not prompt_file.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return text



def parse_reference_image(spec: str) -> tuple[Path, str]:
    """Parse a --reference-image value of the form PATH:TYPE."""
    path, separator, reference_type = spec.rpartition(":")
    if not separator or not path:
        raise ValueError(
            f"Reference image must be given as PATH:TYPE, got: {spec}"
        )
    if reference_type not in REFERENCE_TYPES:
        raise ValueError(
            f"Unknown reference type {reference_type!r} in {spec}. "
            f"Use one of: {', '.join(REFERENCE_TYPES)}."
        )
    return Path(path), reference_type


def load_image(path: Path) -> types.Image:
    """Read an image from disk into the SDK's Image type."""
    mime_type = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(
            f"Unsupported image type: {path}. "
            f"Veo accepts image/png and image/jpeg."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return types.Image(image_bytes=path.read_bytes(), mime_type=mime_type)


def validate_inputs(*, image: Path | None, last_frame: Path | None,
                     reference_images: list[tuple[Path, str]], duration: int,
                     resolution: str) -> None:
    """Reject input combinations the API documents as invalid.

    Checked locally so a bad combination costs nothing instead of a failed
    generation request.
    """
    if last_frame is not None and image is None:
        raise ValueError("--last-frame requires --image; it sets the end of the "
                         "clip that --image starts.")
    if reference_images and image is not None:
        raise ValueError("--reference-image and --image are mutually exclusive.")
    if len(reference_images) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"At most three reference images are supported, got "
            f"{len(reference_images)}."
        )
    if duration != 8 and resolution != "720p":
        raise ValueError(
            f"{resolution} requires --duration 8, got {duration}."
        )
    if duration != 8 and reference_images:
        raise ValueError(
            f"Reference images require --duration 8, got {duration}."
        )
    if duration != 8 and last_frame is not None:
        # Not in Google's docs, which list only extension, reference images
        # and 1080p/4k as requiring 8s. Established by probing the live API:
        # interpolation at 4s returns "Your use case is currently not
        # supported."
        raise ValueError(
            f"--last-frame requires --duration 8, got {duration}."
        )


class LoopOptions(NamedTuple):
    """Values that --loop overrides, or the originals when it is off."""

    last_frame: Path | None
    negative_prompt: str | None


def resolve_loop_options(*, loop: bool, image: Path | None,
                          last_frame: Path | None,
                          negative_prompt: str | None) -> LoopOptions:
    """Apply the --loop preset: end on the starting frame, suppress drift."""
    if not loop:
        return LoopOptions(last_frame, negative_prompt)

    if image is None:
        raise ValueError("--loop requires --image to loop back to.")
    if last_frame is not None:
        raise ValueError("--loop sets the last frame itself; drop --last-frame.")

    return LoopOptions(image, negative_prompt or LOOP_NEGATIVE_PROMPT)


CROSSFADE_DEFAULT_SECONDS = 0.5


class CrossfadeError(Exception):
    """Raised when the post-generation crossfade cannot be applied."""


def validate_crossfade(*, crossfade: float | None, loop: bool,
                        duration: int) -> None:
    """Reject crossfade values that cannot produce a tighter loop."""
    if crossfade is None:
        return
    if not loop:
        raise ValueError("--crossfade tightens the seam that --loop creates, "
                         "so it requires --loop.")
    if crossfade <= 0:
        raise ValueError(
            f"--crossfade must be greater than zero, got {crossfade}."
        )
    if crossfade >= duration:
        raise ValueError(
            f"--crossfade must be shorter than the clip; got {crossfade} for "
            f"a {duration}s clip."
        )


def format_seconds(value: float) -> str:
    """Render a duration for a filtergraph without trailing zeros."""
    return f"{value:g}"


def build_crossfade_filter(*, clip_seconds: float, fade_seconds: float,
                            has_audio: bool) -> str:
    """Build the filtergraph that dissolves the clip's tail over its head.

    --loop ends on the frame the clip started from, but Veo's final frame
    drifts from that anchor, so the wrap still shows a step. Dissolving the
    last `fade_seconds` over the opening `fade_seconds` hides it, at the cost
    of making the clip `fade_seconds` shorter.
    """
    fade = format_seconds(fade_seconds)
    keep = format_seconds(clip_seconds - fade_seconds)
    end = format_seconds(clip_seconds)

    graph = (
        f"[0:v]trim=0:{keep},setpts=PTS-STARTPTS[main];"
        f"[0:v]trim={keep}:{end},setpts=PTS-STARTPTS[tail];"
        f"[main]split[m1][m2];"
        f"[m1]trim=0:{fade},setpts=PTS-STARTPTS[head];"
        f"[m2]trim=start={fade},setpts=PTS-STARTPTS[rest];"
        # A is the head, B the tail. B dominates at T=0 so the wrap matches
        # the frame the clip ended on, A by the time the fade is over.
        f"[head][tail]blend=all_expr='A*(T/{fade})+B*(1-(T/{fade}))'[bl];"
        f"[bl][rest]concat=n=2:v=1:a=0[vout]"
    )
    if not has_audio:
        return graph

    return graph + (
        f";[0:a]atrim=0:{keep},asetpts=PTS-STARTPTS[amain];"
        f"[0:a]atrim={keep}:{end},asetpts=PTS-STARTPTS,"
        f"afade=t=out:st=0:d={fade}[atail];"
        f"[amain]asplit[a1][a2];"
        f"[a1]atrim=0:{fade},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={fade}[ahead];"
        f"[a2]atrim=start={fade},asetpts=PTS-STARTPTS[arest];"
        # normalize=0: the two ramped halves already sum to full level.
        f"[ahead][atail]amix=inputs=2:normalize=0[abl];"
        f"[abl][arest]concat=n=2:v=0:a=1[aout]"
    )


def require_binary(name: str) -> str:
    """Return the path to an ffmpeg-suite binary, or explain how to get it."""
    path = shutil.which(name)
    if path is None:
        raise CrossfadeError(
            f"{name} was not found on PATH; --crossfade needs it. "
            f"Install it with 'brew install ffmpeg'."
        )
    return path


def has_audio_stream(path: str) -> bool:
    """Whether the file carries an audio stream, per ffprobe.

    Veo returns audio, but a clip that lacks it would fail the graph on an
    unresolved [0:a] label, so the audio half is built only when it applies.
    """
    result = subprocess.run(
        [require_binary("ffprobe"), "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise CrossfadeError(
            f"ffprobe could not read {path}: {result.stderr.strip()}"
        )
    return bool(result.stdout.strip())


def apply_crossfade(path: str, *, clip_seconds: float,
                     fade_seconds: float) -> None:
    """Rewrite the clip at `path` with its loop seam crossfaded.

    The faded clip replaces the original only once ffmpeg succeeds, so a
    failed fade leaves the generated video untouched.
    """
    ffmpeg = require_binary("ffmpeg")
    has_audio = has_audio_stream(path)

    source = Path(path)
    target = source.with_suffix(f".crossfade{source.suffix}")

    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
        "-filter_complex", build_crossfade_filter(
            clip_seconds=clip_seconds, fade_seconds=fade_seconds,
            has_audio=has_audio,
        ),
        "-map", "[vout]",
    ]
    if has_audio:
        command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    command += [
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise CrossfadeError(f"ffmpeg failed: {result.stderr.strip()}")
        target.replace(source)
    finally:
        target.unlink(missing_ok=True)


def build_config(*, aspect_ratio: str, resolution: str, duration: int,
                  negative_prompt: str | None = None,
                  person_generation: str | None = None,
                  last_frame: types.Image | None = None,
                  reference_images: list[tuple[types.Image, str]] | None = None,
                  ) -> types.GenerateVideosConfig:
    """Map resolved CLI values onto the SDK config. Unset fields stay None so
    the API applies its own defaults.

    `seed`, `generate_audio`, `fps`, `output_gcs_uri`, `pubsub_topic`, `mask`,
    `compression_quality`, `labels` and `resize_mode` are deliberately absent:
    the SDK rejects them outside Vertex AI, and this script talks to the Gemini
    Developer API. `enhance_prompt` is absent because veo-3.1-generate-preview
    rejects it outright ("`enhancePrompt` isn't supported by this model").
    """
    wrapped_references = [
        types.VideoGenerationReferenceImage(image=image, reference_type=reference_type)
        for image, reference_type in (reference_images or [])
    ] or None

    return types.GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        duration_seconds=duration,
        negative_prompt=negative_prompt,
        person_generation=person_generation,
        last_frame=last_frame,
        reference_images=wrapped_references,
    )


def redact_image_bytes(payload):
    """Copy a serialised config with image payloads replaced by their size.

    A real frame is megabytes of base64; printing it would bury everything
    else in the dry-run output.
    """
    if isinstance(payload, dict):
        return {
            key: (f"<{len(value)} base64 chars>"
                  if key == "image_bytes" and isinstance(value, str)
                  else redact_image_bytes(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [redact_image_bytes(item) for item in payload]
    return payload


def is_transient_error(exc: BaseException) -> bool:
    """Whether a failed request is worth retrying.

    Server errors and rate limits are; a rejected argument or a bad key will
    be rejected just as fast on the fifth attempt as on the first.
    """
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.APIError):
        return exc.code == 429
    return False


def build_client() -> genai.Client:
    # Reads GEMINI_API_KEY from the environment automatically.
    # Pass api_key="..." explicitly if you'd rather not use an env var.
    return genai.Client()


@retry(
    retry=retry_if_exception(is_transient_error),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def start_generation(client: genai.Client, prompt: str, model: str,
                      config: types.GenerateVideosConfig,
                      image: types.Image | None = None):
    """Kick off the long-running video generation operation, with retries
    for transient errors (rate limits, temporary server issues)."""
    return client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=image,
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
    parser.add_argument("prompt", nargs="?",
                         help="Text prompt describing the video")
    parser.add_argument("--prompt-file", "-f", type=Path,
                         help="Read the prompt from this text file instead")
    parser.add_argument("--negative-prompt",
                         help="Describe what should not appear (e.g. "
                              "'cars, traffic'), not 'no cars'")
    parser.add_argument("--model", default="veo-3.1-generate-preview",
                         help="Veo model ID (default: veo-3.1-generate-preview)")
    parser.add_argument("--image", type=Path,
                         help="Image to use as the first frame")
    parser.add_argument("--last-frame", type=Path,
                         help="Image to end on; requires --image")
    parser.add_argument("--reference-image", action="append", default=[],
                         metavar="PATH:TYPE",
                         help="Reference image as PATH:asset or PATH:style. "
                              "Repeatable, up to three. Cannot be combined "
                              "with --image")
    parser.add_argument("--loop", action="store_true",
                         help="End on the starting frame to make the clip "
                              "loop; requires --image")
    parser.add_argument("--crossfade", type=float, nargs="?",
                         const=CROSSFADE_DEFAULT_SECONDS, metavar="SECONDS",
                         help="After generating, dissolve this many seconds of "
                              "the clip's tail over its head to hide the loop "
                              "seam (default: "
                              f"{CROSSFADE_DEFAULT_SECONDS}). Shortens the clip "
                              "by that much; requires --loop")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"],
                         help="Landscape (16:9) or portrait (9:16)")
    parser.add_argument("--resolution", default="1080p",
                         choices=["720p", "1080p", "4k"],
                         help="Output resolution")
    parser.add_argument("--duration", type=int, default=8, choices=[4, 6, 8],
                         help="Clip length in seconds. Must be 8 above 720p "
                              "or with reference images")
    parser.add_argument("--person-generation",
                         choices=["dont_allow", "allow_adult", "allow_all"],
                         help="Whether people may be generated")
    parser.add_argument("--output", default="output.mp4",
                         help="File path to save the generated video to")
    parser.add_argument("--poll-seconds", type=int, default=10,
                         help="Seconds between status checks")
    parser.add_argument("--timeout-seconds", type=int, default=600,
                         help="Max seconds to wait before giving up")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the resolved request and exit without "
                              "calling the API")
    args = parser.parse_args()

    try:
        prompt = resolve_prompt(args.prompt, args.prompt_file)
        reference_images = [parse_reference_image(spec)
                            for spec in args.reference_image]
        loop_options = resolve_loop_options(
            loop=args.loop,
            image=args.image,
            last_frame=args.last_frame,
            negative_prompt=args.negative_prompt,
        )
        validate_crossfade(
            crossfade=args.crossfade,
            loop=args.loop,
            duration=args.duration,
        )
        validate_inputs(
            image=args.image,
            last_frame=loop_options.last_frame,
            reference_images=reference_images,
            duration=args.duration,
            resolution=args.resolution,
        )
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    try:
        image = load_image(args.image) if args.image else None
        last_frame = (load_image(loop_options.last_frame)
                      if loop_options.last_frame else None)
        loaded_references = [(load_image(path), reference_type)
                             for path, reference_type in reference_images]
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    config = build_config(
        aspect_ratio=args.aspect_ratio,
        resolution=args.resolution,
        duration=args.duration,
        negative_prompt=loop_options.negative_prompt,
        person_generation=args.person_generation,
        last_frame=last_frame,
        reference_images=loaded_references,
    )

    if args.dry_run:
        print(f"model: {args.model}")
        print(f"prompt: {prompt}")
        print(f"image: {args.image or '(none)'}")
        if args.crossfade:
            print(f"crossfade: {args.crossfade}s applied after download")
        # exclude_none keeps the output to what is actually being sent.
        payload = redact_image_bytes(config.model_dump(mode="json", exclude_none=True))
        print(json.dumps(payload, indent=2))
        return

    try:
        load_env_file(ENV_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    client = build_client()

    print(f"Starting generation with {args.model}...", file=sys.stderr)
    try:
        operation = start_generation(client, prompt, args.model, config, image)
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

    if args.crossfade:
        print(f"Crossfading {args.crossfade}s to close the loop seam...",
              file=sys.stderr)
        try:
            for path in saved_paths:
                apply_crossfade(path, clip_seconds=args.duration,
                                fade_seconds=args.crossfade)
        except CrossfadeError as e:
            # The clips are already downloaded and intact; a failed fade is
            # worth reporting but not worth discarding a paid generation over.
            print(f"Error: {e}", file=sys.stderr)
            for path in saved_paths:
                print(f"Saved without crossfade: {path}")
            sys.exit(1)

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
