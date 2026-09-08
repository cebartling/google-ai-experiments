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
from google.genai import errors

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


# --- parse_reference_image -------------------------------------------------


def test_parse_reference_image_splits_path_and_type():
    assert generate_video.parse_reference_image("dress.png:asset") == (
        Path("dress.png"), "asset"
    )


def test_parse_reference_image_keeps_directories_in_the_path():
    assert generate_video.parse_reference_image("refs/a/style.jpg:style") == (
        Path("refs/a/style.jpg"), "style"
    )


def test_parse_reference_image_requires_a_type_suffix():
    with pytest.raises(ValueError, match="PATH:TYPE"):
        generate_video.parse_reference_image("dress.png")


def test_parse_reference_image_rejects_unknown_type():
    with pytest.raises(ValueError, match="asset"):
        generate_video.parse_reference_image("dress.png:character")


# --- load_image ------------------------------------------------------------


def test_load_image_reads_bytes_and_infers_png_mime_type(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"\x89PNG fake bytes")

    image = generate_video.load_image(path)

    assert image.image_bytes == b"\x89PNG fake bytes"
    assert image.mime_type == "image/png"


@pytest.mark.parametrize("name", ["frame.jpg", "frame.jpeg", "frame.JPEG"])
def test_load_image_infers_jpeg_mime_type(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"fake jpeg")

    assert generate_video.load_image(path).mime_type == "image/jpeg"


def test_load_image_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "frame.webp"
    path.write_bytes(b"fake webp")

    with pytest.raises(ValueError, match="image/png"):
        generate_video.load_image(path)


def test_load_image_raises_when_file_is_missing(tmp_path):
    missing = tmp_path / "nope.png"

    with pytest.raises(FileNotFoundError) as excinfo:
        generate_video.load_image(missing)

    assert str(missing) in str(excinfo.value)


# --- validate_inputs -------------------------------------------------------


def valid_inputs(**overrides):
    """Baseline arguments that validate_inputs accepts."""
    inputs = dict(
        image=None,
        last_frame=None,
        reference_images=[],
        duration=8,
        resolution="1080p",
    )
    inputs.update(overrides)
    return inputs


def test_validate_inputs_accepts_the_defaults():
    generate_video.validate_inputs(**valid_inputs())


def test_validate_inputs_accepts_interpolation_with_both_frames():
    generate_video.validate_inputs(
        **valid_inputs(image=Path("a.png"), last_frame=Path("b.png"))
    )


def test_validate_inputs_rejects_last_frame_without_image():
    with pytest.raises(ValueError, match="--image"):
        generate_video.validate_inputs(**valid_inputs(last_frame=Path("b.png")))


def test_validate_inputs_rejects_reference_images_with_image():
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_video.validate_inputs(**valid_inputs(
            image=Path("a.png"),
            reference_images=[(Path("r.png"), "asset")],
        ))


def test_validate_inputs_rejects_more_than_three_reference_images():
    refs = [(Path(f"r{i}.png"), "asset") for i in range(4)]

    with pytest.raises(ValueError, match="three"):
        generate_video.validate_inputs(**valid_inputs(reference_images=refs))


@pytest.mark.parametrize("resolution", ["1080p", "4k"])
def test_validate_inputs_requires_eight_seconds_above_720p(resolution):
    with pytest.raises(ValueError, match="8"):
        generate_video.validate_inputs(
            **valid_inputs(duration=4, resolution=resolution)
        )


def test_validate_inputs_allows_short_durations_at_720p():
    generate_video.validate_inputs(**valid_inputs(duration=4, resolution="720p"))


def test_validate_inputs_requires_eight_seconds_with_reference_images():
    with pytest.raises(ValueError, match="8"):
        generate_video.validate_inputs(**valid_inputs(
            duration=4,
            resolution="720p",
            reference_images=[(Path("r.png"), "asset")],
        ))


def test_validate_inputs_requires_eight_seconds_for_interpolation():
    with pytest.raises(ValueError, match="8"):
        generate_video.validate_inputs(**valid_inputs(
            image=Path("a.png"), last_frame=Path("a.png"),
            duration=4, resolution="720p",
        ))


def test_validate_inputs_allows_interpolation_at_eight_seconds():
    generate_video.validate_inputs(**valid_inputs(
        image=Path("a.png"), last_frame=Path("a.png"),
        duration=8, resolution="720p",
    ))


# --- resolve_loop_options --------------------------------------------------


def test_resolve_loop_options_passes_values_through_when_not_looping():
    options = generate_video.resolve_loop_options(
        loop=False, image=Path("a.png"), last_frame=Path("b.png"),
        negative_prompt="rain",
    )

    assert options.last_frame == Path("b.png")
    assert options.negative_prompt == "rain"


def test_resolve_loop_options_reuses_the_start_image_as_the_last_frame():
    options = generate_video.resolve_loop_options(
        loop=True, image=Path("a.png"), last_frame=None, negative_prompt=None,
    )

    assert options.last_frame == Path("a.png")
    assert options.negative_prompt == generate_video.LOOP_NEGATIVE_PROMPT


def test_resolve_loop_options_keeps_an_explicit_negative_prompt():
    options = generate_video.resolve_loop_options(
        loop=True, image=Path("a.png"), last_frame=None,
        negative_prompt="people, text",
    )

    assert options.negative_prompt == "people, text"


def test_resolve_loop_options_requires_a_start_image():
    with pytest.raises(ValueError, match="--image"):
        generate_video.resolve_loop_options(
            loop=True, image=None, last_frame=None, negative_prompt=None,
        )


def test_resolve_loop_options_rejects_an_explicit_last_frame():
    with pytest.raises(ValueError, match="--last-frame"):
        generate_video.resolve_loop_options(
            loop=True, image=Path("a.png"), last_frame=Path("b.png"),
            negative_prompt=None,
        )


# --- build_config ----------------------------------------------------------


def test_build_config_sets_only_the_supplied_fields():
    config = generate_video.build_config(
        aspect_ratio="16:9", resolution="1080p", duration=8,
    )

    assert config.aspect_ratio == "16:9"
    assert config.resolution == "1080p"
    assert config.duration_seconds == 8
    assert config.negative_prompt is None
    assert config.person_generation is None
    assert config.last_frame is None
    assert config.reference_images is None
    # Never set: the API rejects these on the Gemini Developer API.
    assert config.seed is None
    assert config.enhance_prompt is None
    assert config.generate_audio is None


def test_build_config_maps_every_optional_field():
    last_frame = generate_video.types.Image(
        image_bytes=b"end", mime_type="image/png"
    )

    config = generate_video.build_config(
        aspect_ratio="9:16", resolution="720p", duration=8,
        negative_prompt="rain", person_generation="dont_allow",
        last_frame=last_frame,
    )

    assert config.negative_prompt == "rain"
    assert config.person_generation == "dont_allow"
    assert config.last_frame is last_frame


def test_build_config_wraps_reference_images_with_their_types():
    asset = generate_video.types.Image(image_bytes=b"a", mime_type="image/png")
    style = generate_video.types.Image(image_bytes=b"s", mime_type="image/png")

    config = generate_video.build_config(
        aspect_ratio="16:9", resolution="1080p", duration=8,
        reference_images=[(asset, "asset"), (style, "style")],
    )

    # The SDK normalises the reference type onto its enum.
    assert [r.reference_type for r in config.reference_images] == [
        generate_video.types.VideoGenerationReferenceType.ASSET,
        generate_video.types.VideoGenerationReferenceType.STYLE,
    ]
    assert config.reference_images[0].image is asset


# --- is_transient_error ----------------------------------------------------


def test_is_transient_error_retries_server_errors():
    assert generate_video.is_transient_error(errors.ServerError(503, {})) is True


def test_is_transient_error_retries_rate_limits():
    assert generate_video.is_transient_error(errors.ClientError(429, {})) is True


def test_is_transient_error_does_not_retry_bad_requests():
    assert generate_video.is_transient_error(errors.ClientError(400, {})) is False


def test_is_transient_error_does_not_retry_arbitrary_exceptions():
    assert generate_video.is_transient_error(ValueError("bad flag")) is False


# --- redact_image_bytes ----------------------------------------------------


def test_redact_image_bytes_replaces_nested_payloads():
    payload = {
        "resolution": "720p",
        "last_frame": {"image_bytes": "aGVsbG8=", "mime_type": "image/png"},
        "reference_images": [
            {"image": {"image_bytes": "aGk=", "mime_type": "image/png"}},
        ],
    }

    redacted = generate_video.redact_image_bytes(payload)

    assert redacted["resolution"] == "720p"
    assert redacted["last_frame"]["image_bytes"] == "<8 base64 chars>"
    assert redacted["last_frame"]["mime_type"] == "image/png"
    assert redacted["reference_images"][0]["image"]["image_bytes"] == (
        "<4 base64 chars>"
    )


def test_redact_image_bytes_leaves_the_original_untouched():
    payload = {"last_frame": {"image_bytes": "aGVsbG8="}}

    generate_video.redact_image_bytes(payload)

    assert payload["last_frame"]["image_bytes"] == "aGVsbG8="

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --- validate_crossfade ----------------------------------------------------


def test_validate_crossfade_accepts_none():
    generate_video.validate_crossfade(crossfade=None, loop=False, duration=8)


def test_validate_crossfade_accepts_a_fade_shorter_than_the_clip():
    generate_video.validate_crossfade(crossfade=0.5, loop=True, duration=8)


def test_validate_crossfade_requires_loop():
    with pytest.raises(ValueError, match="--loop"):
        generate_video.validate_crossfade(crossfade=0.5, loop=False, duration=8)


def test_validate_crossfade_rejects_a_non_positive_fade():
    with pytest.raises(ValueError, match="greater than zero"):
        generate_video.validate_crossfade(crossfade=0, loop=True, duration=8)


def test_validate_crossfade_rejects_a_fade_as_long_as_the_clip():
    with pytest.raises(ValueError, match="shorter than the clip"):
        generate_video.validate_crossfade(crossfade=8, loop=True, duration=8)


# --- build_crossfade_filter ------------------------------------------------


def test_build_crossfade_filter_dissolves_the_tail_over_the_head():
    graph = generate_video.build_crossfade_filter(
        clip_seconds=8, fade_seconds=0.5, has_audio=False,
    )

    # The tail starts where the kept portion ends: 8 - 0.5.
    assert "trim=7.5:8" in graph
    assert "trim=0:7.5" in graph
    # Tail (B) dominates at T=0 and the head (A) at the end of the fade.
    assert "A*(T/0.5)+B*(1-(T/0.5))" in graph
    assert "[vout]" in graph


def test_build_crossfade_filter_omits_audio_when_the_clip_has_none():
    graph = generate_video.build_crossfade_filter(
        clip_seconds=8, fade_seconds=0.5, has_audio=False,
    )

    assert "afade" not in graph
    assert "[aout]" not in graph


def test_build_crossfade_filter_mirrors_the_fade_on_audio():
    graph = generate_video.build_crossfade_filter(
        clip_seconds=8, fade_seconds=0.5, has_audio=True,
    )

    assert "afade=t=in:st=0:d=0.5" in graph
    assert "afade=t=out:st=0:d=0.5" in graph
    # normalize=0 keeps the sum from halving the level through the blend.
    assert "amix=inputs=2:normalize=0" in graph
    assert "[aout]" in graph


# --- apply_crossfade -------------------------------------------------------


def test_apply_crossfade_reports_a_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_video.shutil, "which", lambda name: None)

    with pytest.raises(generate_video.CrossfadeError, match="ffmpeg"):
        generate_video.apply_crossfade(
            str(tmp_path / "clip.mp4"), clip_seconds=8, fade_seconds=0.5,
        )


def test_apply_crossfade_replaces_the_clip_in_place(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original")
    monkeypatch.setattr(generate_video.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(generate_video, "has_audio_stream", lambda path: True)

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        # Stand in for ffmpeg writing its output file.
        Path(command[-1]).write_bytes(b"crossfaded")
        return generate_video.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(generate_video.subprocess, "run", fake_run)

    generate_video.apply_crossfade(str(clip), clip_seconds=8, fade_seconds=0.5)

    assert clip.read_bytes() == b"crossfaded"
    assert commands[0][0] == "/usr/bin/ffmpeg"
    # The temporary file ffmpeg wrote to is not left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["clip.mp4"]


def test_apply_crossfade_raises_when_ffmpeg_fails(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original")
    monkeypatch.setattr(generate_video.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(generate_video, "has_audio_stream", lambda path: False)
    monkeypatch.setattr(
        generate_video.subprocess, "run",
        lambda command, **kwargs: generate_video.subprocess.CompletedProcess(
            command, 1, "", "Invalid argument"
        ),
    )

    with pytest.raises(generate_video.CrossfadeError, match="Invalid argument"):
        generate_video.apply_crossfade(str(clip), clip_seconds=8, fade_seconds=0.5)

    # The original clip survives a failed fade.
    assert clip.read_bytes() == b"original"


# --- expected_person_generation --------------------------------------------


def test_expected_person_generation_is_allow_all_for_text_to_video():
    assert generate_video.expected_person_generation(
        image=None, reference_images=[],
    ) == "allow_all"


def test_expected_person_generation_is_allow_adult_with_a_start_image():
    assert generate_video.expected_person_generation(
        image=Path("frame.png"), reference_images=[],
    ) == "allow_adult"


def test_expected_person_generation_is_allow_adult_with_reference_images():
    assert generate_video.expected_person_generation(
        image=None, reference_images=[(Path("dress.png"), "asset")],
    ) == "allow_adult"


# --- check_person_generation -----------------------------------------------


def test_check_person_generation_is_quiet_when_unset():
    assert generate_video.check_person_generation(
        person_generation=None, image=None, reference_images=[],
    ) is None


def test_check_person_generation_is_quiet_when_it_matches_the_mode():
    assert generate_video.check_person_generation(
        person_generation="allow_all", image=None, reference_images=[],
    ) is None
    assert generate_video.check_person_generation(
        person_generation="allow_adult", image=Path("a.png"), reference_images=[],
    ) is None


def test_check_person_generation_warns_on_allow_adult_for_text_to_video():
    # The combination that returned "allow_adult for personGeneration is
    # currently not supported" from the live API.
    warning = generate_video.check_person_generation(
        person_generation="allow_adult", image=None, reference_images=[],
    )

    assert warning is not None
    assert "allow_all" in warning


def test_check_person_generation_warns_on_allow_all_for_image_to_video():
    warning = generate_video.check_person_generation(
        person_generation="allow_all", image=Path("a.png"), reference_images=[],
    )

    assert warning is not None
    assert "allow_adult" in warning


def test_check_person_generation_mentions_the_regional_exception():
    # It stays a warning rather than an error because EU/UK/CH/MENA accept
    # only allow_adult, which would make this exact request correct there.
    warning = generate_video.check_person_generation(
        person_generation="allow_adult", image=None, reference_images=[],
    )

    assert "EU" in warning
