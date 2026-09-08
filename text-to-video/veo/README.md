# Veo text-to-video

`generate_video.py` drives Veo through the Gemini API. It is a single
[uv](https://docs.astral.sh/uv/) script — dependencies are declared inline, so
there is no venv to create.

## Requirements

- [uv](https://docs.astral.sh/uv/) — the only thing to install. Dependencies
  (`google-genai`, `tenacity`, `python-dotenv`) are declared in a PEP 723
  header inside the script and resolved on first run.
- Python 3.10 or newer, which uv will fetch if you don't have it.
- A Gemini API key with billing enabled.

## Setup

```bash
cp .env.example .env.local   # then fill in GEMINI_API_KEY
chmod +x generate_video.py
```

The key comes from <https://aistudio.google.com/apikey>, and `.env.local` is
gitignored. The script looks for it next to itself, not in the working
directory, so it runs correctly from anywhere; a missing file is a hard error
before any request is made.

Every generation is billed, so use `--dry-run` while you are still working out
the flags.

## Usage

```bash
./generate_video.py "A drone flies through a canyon at sunset"
./generate_video.py --prompt-file prompt.txt --output canyon.mp4
```

Run the tests with `uv run test_generate_video.py`.

## How a run proceeds

Video generation is a long-running operation, so a run is three phases and
takes minutes, not seconds:

1. **Kick off.** The request returns an operation handle immediately. Transient
   failures — HTTP 5xx and 429 rate limits — are retried up to five times with
   exponential backoff. Anything else (a bad key, a rejected argument) fails
   straight away rather than being retried.
2. **Poll.** The script checks the operation every `--poll-seconds` and prints
   elapsed time to stderr, giving up at `--timeout-seconds`. Google documents
   latency of 11 seconds to 6 minutes, so the 600 s default has room but a
   generation that hits the ceiling during peak hours will need it raised.
3. **Download.** The finished video is fetched and written to `--output`, which
   is **overwritten without warning** if it already exists.

Progress goes to stderr and the saved path to stdout, so `--output` paths can be
piped somewhere useful.

## Flags

| Flag | Purpose |
| --- | --- |
| `prompt` / `--prompt-file` | The prompt, inline or from a file |
| `--negative-prompt` | What should *not* appear |
| `--image` | Still to use as the first frame |
| `--last-frame` | Still to end on; requires `--image` |
| `--reference-image PATH:TYPE` | Up to three refs, `asset` or `style` |
| `--loop` | End on the starting frame; requires `--image` |
| `--duration {4,6,8}` | Clip length in seconds (default 8) |
| `--person-generation` | `dont_allow`, `allow_adult`, `allow_all` |
| `--aspect-ratio`, `--resolution` | `16:9`/`9:16`, `720p`/`1080p`/`4k` |
| `--dry-run` | Print the resolved request and exit; no API call |
| `--model` | Veo model ID (default `veo-3.1-generate-preview`) |
| `--output` | Where to write the `.mp4` (default `output.mp4`) |
| `--poll-seconds` | Seconds between status checks (default 10) |
| `--timeout-seconds` | Give up after this long (default 600) |

`--help` prints the same list with the exact choices for each flag.

### Negative prompts

Describe what you don't want to see rather than negating it — `"cars, traffic"`,
not `"no cars"`.

```bash
./generate_video.py "a busy city street" --negative-prompt "cars, traffic"
```

### Images

Three separate mechanisms, and they don't mix:

- **`--image`** animates a still as the first frame.
- **`--image` + `--last-frame`** interpolates between two stills.
- **`--reference-image`** carries a character, object, or look into a
  text-to-video generation. `asset` brings the subject through; `style` brings
  the aesthetic. Mutually exclusive with `--image`.

```bash
./generate_video.py "the fog rolls in and she fades away" \
    --image start.png --last-frame end.png

./generate_video.py "a woman walks through a shallow lagoon" \
    --reference-image woman.png:asset --reference-image dress.png:asset
```

Images must be PNG or JPEG.

### Looping

Veo has no loop parameter. `--loop` builds one out of the parts that exist: it
passes `--image` as the last frame too, so the clip ends where it began, and
sets a negative prompt against the drift that would give the seam away
(camera movement, cuts, fades).

**Interpolation is locked to 8 seconds.** Passing `--last-frame` (which `--loop`
does) with `--duration 4` or `6` is rejected by the API as *"Your use case is
currently not supported"*. This is not in Google's documentation — it was
established by probing the live API — so the script now rejects it locally.

```bash
./generate_video.py "a candle flame flickers in a dark room" \
    --image frame.png --loop --duration 8 --resolution 720p
```

What makes the seam hold up:

- Prompt **cyclical motion**, not a narrative arc — rippling water, drifting
  steam, an object completing one full rotation. Anything that ends somewhere
  different from where it started will fight the constraint.
- 8 seconds is a lot of time to drift, and you cannot ask for less. Favour
  scenes whose motion is genuinely repetitive over a long beat.
- **Strip the audio afterwards** — it cannot be disabled at generation time
  (see below) and does not loop cleanly even when the picture does:
  `ffmpeg -i in.mp4 -c copy -an out.mp4`.
- Supply your own `--negative-prompt` to override the built-in one if your
  scene needs different suppressions — `--loop` only fills it in when you
  haven't.

Verify a seam objectively rather than trusting your eye — compare the last
frame against the first, and check that difference against two adjacent
mid-clip frames as a baseline:

```bash
ffmpeg -i last.png -i first.png -lavfi psnr -f null -
```

If the seam scores materially *worse* than the adjacent-frame baseline, it will
read as a visible jump.

Longer loops are not directly supported. The API's video-extension feature
(+7 s per call) is the other lever, and is not wired into this script.

## Constraints the API enforces

These are checked locally, before any request is sent:

| Rule | |
| --- | --- |
| `--last-frame` | Only valid alongside `--image` |
| `--reference-image` | Max 3; cannot combine with `--image` |
| `--duration` | Must be `8` at `1080p` or `4k`, with reference images, and with `--last-frame` |

Also worth knowing: output is 24 fps, one video per request, and generated
videos are deleted from the server after **2 days** — download promptly. All
output is SynthID-watermarked.

Model support varies (`4k` is Veo 3.1 / 3.1 Fast only, not Lite or Veo 3), and
`--model` takes an arbitrary string, so that combination is left to the API to
validate.


## Troubleshooting

**"Environment file not found"** — `.env.local` doesn't exist next to the
script. Copy `.env.example` onto it.

**A flag combination is rejected before anything happens.** That is the local
validation doing its job; the message names the rule. Nothing was billed.

**Generations are not reproducible.** There is no seed. `seed` exists in the
API but the SDK rejects it outside Vertex AI, so repeated runs of an identical
command return different videos.

**A generation was blocked.** Veo runs safety filters over the audio as well as
the video and will sometimes block on the audio alone. Blocked generations are
not charged.

**The video URL 404s a few days later.** Generated videos are deleted from the
server after 2 days. The script downloads immediately, so this only bites if
you are reusing an old operation.

## Not available on this API surface

The script talks to the **Gemini Developer API** (an AI Studio key). These
`GenerateVideosConfig` fields exist in the SDK but are rejected outside Vertex
AI, so the script does not expose them: `seed`, `generate_audio`, `fps`,
`output_gcs_uri`, `pubsub_topic`, `mask`, `compression_quality`, `labels`,
`resize_mode`.

Separately, `enhance_prompt` is rejected by `veo-3.1-generate-preview` itself
(*"`enhancePrompt` isn't supported by this model"*), so it is never set.

The practical consequences: **no reproducible seeds**, and **audio cannot be
turned off at generation time** — strip it afterwards with
`ffmpeg -i in.mp4 -c copy -an out.mp4`.
