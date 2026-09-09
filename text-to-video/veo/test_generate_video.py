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

import json
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


# --- estimate_cost_usd -----------------------------------------------------

def test_estimate_cost_usd_prices_standard_veo_at_forty_cents_a_second():
    cost = generate_video.estimate_cost_usd(
        model="veo-3.1-generate-preview", resolution="1080p", duration=8,
    )

    assert cost == pytest.approx(3.20)


def test_estimate_cost_usd_charges_more_for_standard_at_4k():
    cost = generate_video.estimate_cost_usd(
        model="veo-3.1-generate-preview", resolution="4k", duration=8,
    )

    assert cost == pytest.approx(4.80)


@pytest.mark.parametrize("resolution,expected", [
    ("720p", 0.80),
    ("1080p", 0.96),
    ("4k", 2.40),
])
def test_estimate_cost_usd_prices_fast_by_resolution(resolution, expected):
    cost = generate_video.estimate_cost_usd(
        model="veo-3.1-fast-generate-preview", resolution=resolution, duration=8,
    )

    assert cost == pytest.approx(expected)


def test_estimate_cost_usd_prices_lite_below_fast():
    cost = generate_video.estimate_cost_usd(
        model="veo-3.1-lite-generate-preview", resolution="720p", duration=8,
    )

    assert cost == pytest.approx(0.40)


def test_estimate_cost_usd_scales_with_duration():
    cost = generate_video.estimate_cost_usd(
        model="veo-3.1-lite-generate-preview", resolution="720p", duration=4,
    )

    assert cost == pytest.approx(0.20)


def test_estimate_cost_usd_returns_none_for_an_unknown_model():
    assert generate_video.estimate_cost_usd(
        model="veo-9-imaginary", resolution="1080p", duration=8,
    ) is None


def test_estimate_cost_usd_returns_none_for_a_resolution_the_model_does_not_support():
    assert generate_video.estimate_cost_usd(
        model="veo-3.1-lite-generate-preview", resolution="4k", duration=8,
    ) is None


# --- format_wait -----------------------------------------------------------

def test_format_wait_renders_minutes_and_seconds():
    assert generate_video.format_wait(372) == "6m12s"


def test_format_wait_omits_minutes_under_a_minute():
    assert generate_video.format_wait(45) == "45s"


def test_format_wait_rounds_a_partial_second_up():
    assert generate_video.format_wait(44.2) == "45s"


def test_format_wait_never_reports_zero_for_a_real_wait():
    assert generate_video.format_wait(0.1) == "1s"


# --- resolve_spend_limit_usd -----------------------------------------------

def test_resolve_spend_limit_usd_defaults_to_ten_dollars():
    assert generate_video.resolve_spend_limit_usd(None) == pytest.approx(10.00)


def test_resolve_spend_limit_usd_reads_the_environment_override():
    assert generate_video.resolve_spend_limit_usd("2.50") == pytest.approx(2.50)


def test_resolve_spend_limit_usd_ignores_a_blank_value():
    assert generate_video.resolve_spend_limit_usd("  ") == pytest.approx(10.00)


def test_resolve_spend_limit_usd_rejects_a_non_numeric_value():
    with pytest.raises(ValueError, match="VEO_SPEND_LIMIT_USD"):
        generate_video.resolve_spend_limit_usd("ten dollars")


def test_resolve_spend_limit_usd_rejects_a_non_positive_value():
    with pytest.raises(ValueError, match="VEO_SPEND_LIMIT_USD"):
        generate_video.resolve_spend_limit_usd("0")


# --- read_ledger -----------------------------------------------------------

def test_read_ledger_returns_no_entries_when_the_file_is_missing(tmp_path):
    assert generate_video.read_ledger(tmp_path / "absent.json") == []


def test_read_ledger_reads_recorded_entries(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([
        {"timestamp": 1000.0, "usd": 3.20, "model": "veo-3.1-generate-preview"},
    ]))

    entries = generate_video.read_ledger(ledger)

    assert len(entries) == 1
    assert entries[0]["timestamp"] == pytest.approx(1000.0)
    assert entries[0]["usd"] == pytest.approx(3.20)


def test_read_ledger_ignores_a_corrupt_file(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{not json")

    assert generate_video.read_ledger(ledger) == []


def test_read_ledger_ignores_a_file_that_is_not_a_list(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"usd": 1}')

    assert generate_video.read_ledger(ledger) == []


def test_read_ledger_drops_entries_missing_a_timestamp_or_amount(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([
        {"timestamp": 1000.0, "usd": 3.20},
        {"timestamp": 1001.0},
        {"usd": 1.0},
        "not a dict",
    ]))

    entries = generate_video.read_ledger(ledger)

    assert [e["timestamp"] for e in entries] == [pytest.approx(1000.0)]


# --- prune_entries ---------------------------------------------------------

def test_prune_entries_keeps_entries_inside_the_window():
    entries = [{"timestamp": 500.0, "usd": 1.0}]

    assert generate_video.prune_entries(entries, now=1000.0) == entries


def test_prune_entries_drops_entries_older_than_the_window():
    entries = [{"timestamp": 399.0, "usd": 1.0}]

    assert generate_video.prune_entries(entries, now=1000.0) == []


def test_prune_entries_drops_an_entry_exactly_at_the_window_edge():
    entries = [{"timestamp": 400.0, "usd": 1.0}]

    assert generate_video.prune_entries(entries, now=1000.0) == []


# --- record_spend ----------------------------------------------------------

def test_record_spend_creates_the_ledger_with_one_entry(tmp_path):
    ledger = tmp_path / ".spend_ledger.json"

    generate_video.record_spend(ledger, usd=3.20, model="veo-3.1", now=1000.0)

    entries = generate_video.read_ledger(ledger)
    assert len(entries) == 1
    assert entries[0]["timestamp"] == pytest.approx(1000.0)
    assert entries[0]["usd"] == pytest.approx(3.20)
    assert entries[0]["model"] == "veo-3.1"


def test_record_spend_appends_to_existing_entries(tmp_path):
    ledger = tmp_path / ".spend_ledger.json"
    generate_video.record_spend(ledger, usd=3.20, model="veo-3.1", now=1000.0)

    generate_video.record_spend(ledger, usd=0.40, model="veo-3.1", now=1100.0)

    amounts = [e["usd"] for e in generate_video.read_ledger(ledger)]
    assert amounts == [pytest.approx(3.20), pytest.approx(0.40)]


def test_record_spend_prunes_entries_from_before_today(tmp_path):
    # The ledger is retained to the last Pacific midnight, not to the spend
    # window, because the daily request count needs the whole day.
    ledger = tmp_path / ".spend_ledger.json"
    generate_video.record_spend(ledger, usd=3.20, model="veo-3.1",
                                now=1788850500.0)  # 2026-09-07 23:55 PDT

    generate_video.record_spend(ledger, usd=0.40, model="veo-3.1",
                                now=1788921420.0)  # 2026-09-08 19:37 PDT

    amounts = [e["usd"] for e in generate_video.read_ledger(ledger)]
    assert amounts == [pytest.approx(0.40)]


def test_record_spend_keeps_entries_from_earlier_today(tmp_path):
    ledger = tmp_path / ".spend_ledger.json"
    generate_video.record_spend(ledger, usd=3.20, model="veo-3.1",
                                now=1788850800.0 + 3600)  # 01:00 PDT

    generate_video.record_spend(ledger, usd=0.40, model="veo-3.1",
                                now=1788921420.0)         # 19:37 PDT, same day

    amounts = [e["usd"] for e in generate_video.read_ledger(ledger)]
    assert amounts == [pytest.approx(3.20), pytest.approx(0.40)]


def test_record_spend_leaves_no_temporary_file_behind(tmp_path):
    ledger = tmp_path / ".spend_ledger.json"

    generate_video.record_spend(ledger, usd=3.20, model="veo-3.1", now=1000.0)

    assert [p.name for p in tmp_path.iterdir()] == [".spend_ledger.json"]


def test_record_spend_replaces_a_corrupt_ledger_rather_than_raising(tmp_path):
    ledger = tmp_path / ".spend_ledger.json"
    ledger.write_text("{not json")

    generate_video.record_spend(ledger, usd=3.20, model="veo-3.1", now=1000.0)

    assert len(generate_video.read_ledger(ledger)) == 1


def test_record_spend_is_silent_when_the_ledger_cannot_be_written(tmp_path):
    unwritable = tmp_path / "no-such-dir" / ".spend_ledger.json"

    generate_video.record_spend(unwritable, usd=3.20, model="veo-3.1", now=1000.0)

    assert not unwritable.exists()


# --- wait_seconds_until_affordable -----------------------------------------

def test_wait_seconds_until_affordable_is_zero_when_the_request_fits():
    wait = generate_video.wait_seconds_until_affordable(
        cost_usd=3.20,
        entries=[{"timestamp": 900.0, "usd": 3.20}],
        limit_usd=10.00, now=1000.0,
    )

    assert wait == 0


def test_wait_seconds_until_affordable_is_zero_when_the_request_lands_exactly_on_the_limit():
    wait = generate_video.wait_seconds_until_affordable(
        cost_usd=0.40,
        entries=[{"timestamp": 900.0, "usd": 9.60}],
        limit_usd=10.00, now=1000.0,
    )

    assert wait == 0


def test_wait_seconds_until_affordable_waits_for_the_oldest_entry_to_age_out():
    wait = generate_video.wait_seconds_until_affordable(
        cost_usd=3.20,
        entries=[{"timestamp": 772.0, "usd": 9.60}],
        limit_usd=10.00, now=1000.0,
    )

    assert wait == pytest.approx(372.0)


def test_wait_seconds_until_affordable_waits_for_several_entries_when_one_is_not_enough():
    wait = generate_video.wait_seconds_until_affordable(
        cost_usd=3.20,
        entries=[
            {"timestamp": 500.0, "usd": 1.00},
            {"timestamp": 772.0, "usd": 8.60},
        ],
        limit_usd=10.00, now=1000.0,
    )

    # Shedding the $1.00 entry is not enough to make room, so the wait runs
    # until the second entry expires too.
    assert wait == pytest.approx(372.0)


def test_wait_seconds_until_affordable_returns_none_when_the_request_alone_exceeds_the_limit():
    wait = generate_video.wait_seconds_until_affordable(
        cost_usd=12.00, entries=[], limit_usd=10.00, now=1000.0,
    )

    assert wait is None


# --- spend_preflight_error -------------------------------------------------

def test_spend_preflight_error_is_none_when_the_request_fits():
    assert generate_video.spend_preflight_error(
        cost_usd=3.20,
        entries=[{"timestamp": 900.0, "usd": 3.20}],
        limit_usd=10.00, now=1000.0,
    ) is None


def test_spend_preflight_error_is_none_at_exactly_the_limit():
    assert generate_video.spend_preflight_error(
        cost_usd=0.40,
        entries=[{"timestamp": 900.0, "usd": 9.60}],
        limit_usd=10.00, now=1000.0,
    ) is None


def test_spend_preflight_error_is_none_when_the_cost_is_unknown():
    assert generate_video.spend_preflight_error(
        cost_usd=None,
        entries=[{"timestamp": 900.0, "usd": 9.60}],
        limit_usd=10.00, now=1000.0,
    ) is None


def test_spend_preflight_error_describes_the_cost_the_spend_and_the_wait():
    message = generate_video.spend_preflight_error(
        cost_usd=3.20,
        entries=[{"timestamp": 772.0, "usd": 9.60}],
        limit_usd=10.00, now=1000.0,
    )

    assert message == (
        "this request costs $3.20; $9.60 already spent in the last 10 "
        "minutes against a $10.00 limit. Wait 6m12s, or pass --force "
        "to send it anyway."
    )


def test_spend_preflight_error_says_waiting_cannot_help_when_the_cost_exceeds_the_limit():
    message = generate_video.spend_preflight_error(
        cost_usd=12.00, entries=[], limit_usd=10.00, now=1000.0,
    )

    assert "waiting will not help" in message
    assert "Pass --force to send it anyway." in message
    assert "--force-spend" not in message


# --- is_daily_quota_error --------------------------------------------------

def rate_limit_error(*violations):
    """A 429 carrying the QuotaFailure detail Google sometimes includes."""
    body = {"error": {"code": 429, "message": "You exceeded your current quota.",
                      "status": "RESOURCE_EXHAUSTED"}}
    if violations:
        body["error"]["details"] = [{
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id} for quota_id in violations],
        }]
    return errors.ClientError(429, body)


def test_is_daily_quota_error_detects_a_per_day_violation():
    exc = rate_limit_error("GenerateRequestsPerDayPerProjectPerModel")

    assert generate_video.is_daily_quota_error(exc) is True


def test_is_daily_quota_error_ignores_a_per_minute_violation():
    exc = rate_limit_error("GenerateRequestsPerMinutePerProjectPerModel")

    assert generate_video.is_daily_quota_error(exc) is False


def test_is_daily_quota_error_detects_a_per_day_violation_among_others():
    exc = rate_limit_error("GenerateRequestsPerMinutePerProjectPerModel",
                           "GenerateRequestsPerDayPerProjectPerModel")

    assert generate_video.is_daily_quota_error(exc) is True


def test_is_daily_quota_error_is_false_when_the_error_names_no_quota():
    assert generate_video.is_daily_quota_error(rate_limit_error()) is False


def test_is_daily_quota_error_is_false_for_a_non_rate_limit_error():
    assert generate_video.is_daily_quota_error(errors.ClientError(400, {})) is False


def test_is_daily_quota_error_is_false_for_an_arbitrary_exception():
    assert generate_video.is_daily_quota_error(ValueError("bad flag")) is False


# --- is_transient_error, daily quotas --------------------------------------

def test_is_transient_error_does_not_retry_an_exhausted_daily_quota():
    exc = rate_limit_error("GenerateRequestsPerDayPerProjectPerModel")

    assert generate_video.is_transient_error(exc) is False


def test_is_transient_error_still_retries_a_per_minute_rate_limit():
    exc = rate_limit_error("GenerateRequestsPerMinutePerProjectPerModel")

    assert generate_video.is_transient_error(exc) is True


# --- stop_after_transient_attempts -----------------------------------------

class FakeOutcome:
    def __init__(self, exc):
        self._exc = exc

    def exception(self):
        return self._exc


class FakeRetryState:
    def __init__(self, exc, attempt_number):
        self.outcome = FakeOutcome(exc)
        self.attempt_number = attempt_number


def test_stop_after_transient_attempts_gives_server_errors_five_tries():
    exc = errors.ServerError(503, {})

    assert generate_video.stop_after_transient_attempts(
        FakeRetryState(exc, 4)) is False
    assert generate_video.stop_after_transient_attempts(
        FakeRetryState(exc, 5)) is True


def test_stop_after_transient_attempts_gives_rate_limits_a_shorter_leash():
    exc = rate_limit_error()

    assert generate_video.stop_after_transient_attempts(
        FakeRetryState(exc, 1)) is False
    assert generate_video.stop_after_transient_attempts(
        FakeRetryState(exc, 2)) is True


def test_stop_after_transient_attempts_stops_when_there_is_no_outcome():
    state = FakeRetryState(errors.ServerError(503, {}), 1)
    state.outcome = None

    assert generate_video.stop_after_transient_attempts(state) is False


# --- format_wait, multi-hour waits -----------------------------------------

def test_format_wait_renders_hours_and_minutes():
    assert generate_video.format_wait(22320) == "6h12m"


def test_format_wait_omits_seconds_once_it_reaches_an_hour():
    assert generate_video.format_wait(3600) == "1h0m"


def test_format_wait_still_renders_minutes_and_seconds_under_an_hour():
    assert generate_video.format_wait(3599) == "59m59s"


# --- quota_day_start -------------------------------------------------------

# Fixtures are real epoch seconds in America/Los_Angeles (PDT, UTC-7):
#   1788921420 = 2026-09-08 19:37
#   1788850800 = 2026-09-08 00:00  <- that day's start
#   1788851100 = 2026-09-08 00:05
#   1788850500 = 2026-09-07 23:55
#   1788764400 = 2026-09-07 00:00

def test_quota_day_start_returns_the_most_recent_pacific_midnight():
    assert generate_video.quota_day_start(1788921420.0) == 1788850800.0


def test_quota_day_start_just_after_midnight_returns_that_midnight():
    assert generate_video.quota_day_start(1788851100.0) == 1788850800.0


def test_quota_day_start_just_before_midnight_returns_the_previous_day():
    assert generate_video.quota_day_start(1788850500.0) == 1788764400.0


# --- resolve_daily_request_limit -------------------------------------------

def test_resolve_daily_request_limit_defaults_to_ten():
    assert generate_video.resolve_daily_request_limit(None) == 10


def test_resolve_daily_request_limit_reads_the_environment_override():
    assert generate_video.resolve_daily_request_limit("50") == 50


def test_resolve_daily_request_limit_ignores_a_blank_value():
    assert generate_video.resolve_daily_request_limit("  ") == 10


def test_resolve_daily_request_limit_rejects_a_non_numeric_value():
    with pytest.raises(ValueError, match="VEO_DAILY_REQUEST_LIMIT"):
        generate_video.resolve_daily_request_limit("ten")


def test_resolve_daily_request_limit_rejects_a_non_positive_value():
    with pytest.raises(ValueError, match="VEO_DAILY_REQUEST_LIMIT"):
        generate_video.resolve_daily_request_limit("0")


# --- count_requests_since --------------------------------------------------

def test_count_requests_since_counts_entries_on_or_after_the_boundary():
    entries = [
        {"timestamp": 1788850500.0, "usd": 3.20},   # yesterday, 23:55
        {"timestamp": 1788850800.0, "usd": 3.20},   # exactly midnight
        {"timestamp": 1788851100.0, "usd": 3.20},   # today, 00:05
    ]

    assert generate_video.count_requests_since(entries, since=1788850800.0) == 2


def test_count_requests_since_is_zero_for_an_empty_ledger():
    assert generate_video.count_requests_since([], since=1788850800.0) == 0


# --- prune_for_retention ---------------------------------------------------

def test_prune_for_retention_keeps_entries_from_earlier_today():
    # 06:00 Pacific, hours older than the spend window but still today.
    entries = [{"timestamp": 1788850800.0 + 6 * 3600, "usd": 3.20}]

    assert generate_video.prune_for_retention(entries, now=1788921420.0) == entries


def test_prune_for_retention_keeps_an_entry_exactly_at_midnight():
    # count_requests_since counts the boundary as today, so retention must
    # keep it — otherwise the first request of the day is never counted.
    entries = [{"timestamp": 1788850800.0, "usd": 3.20}]

    assert generate_video.prune_for_retention(entries, now=1788921420.0) == entries


def test_prune_for_retention_drops_entries_from_before_today():
    entries = [{"timestamp": 1788850500.0, "usd": 3.20}]

    assert generate_video.prune_for_retention(entries, now=1788921420.0) == []


def test_prune_for_retention_keeps_yesterdays_entry_still_inside_the_spend_window():
    # 00:02 Pacific: the 10-minute spend window reaches back past midnight, so
    # an entry from 23:55 yesterday still counts against spend and must survive.
    now = 1788850800.0 + 120
    entries = [{"timestamp": 1788850500.0, "usd": 3.20}]

    assert generate_video.prune_for_retention(entries, now=now) == entries


# --- daily_quota_error -----------------------------------------------------

def today_entries(count):
    """`count` requests spread through today, well inside the retention window."""
    return [{"timestamp": 1788850800.0 + 3600 + i, "usd": 3.20}
            for i in range(count)]


def test_daily_quota_error_is_none_with_requests_to_spare():
    assert generate_video.daily_quota_error(
        entries=today_entries(3), limit_requests=10, now=1788921420.0,
    ) is None


def test_daily_quota_error_is_none_on_the_last_available_request():
    assert generate_video.daily_quota_error(
        entries=today_entries(9), limit_requests=10, now=1788921420.0,
    ) is None


def test_daily_quota_error_blocks_once_the_limit_is_reached():
    message = generate_video.daily_quota_error(
        entries=today_entries(10), limit_requests=10, now=1788921420.0,
    )

    assert message == (
        "10 of 10 requests used today; the daily quota resets at midnight "
        "Pacific, in 4h23m. Pass --force to send it anyway."
    )


def test_daily_quota_error_ignores_entries_from_before_today():
    entries = [{"timestamp": 1788850500.0, "usd": 3.20} for _ in range(20)]

    assert generate_video.daily_quota_error(
        entries=entries, limit_requests=10, now=1788921420.0,
    ) is None


def test_daily_quota_error_counts_a_request_made_exactly_at_midnight():
    entries = generate_video.prune_for_retention(
        [{"timestamp": 1788850800.0 + 60 * i, "usd": 3.20} for i in range(10)],
        now=1788921420.0,
    )

    assert generate_video.daily_quota_error(
        entries=entries, limit_requests=10, now=1788921420.0,
    ) is not None
