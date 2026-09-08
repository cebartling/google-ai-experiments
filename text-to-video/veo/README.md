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
| `--seed` | Fixed seed for reproducible output |
| `--person-generation` | `dont_allow`, `allow_adult`, `allow_all` |
| `--no-audio` | Drop the generated soundtrack |
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
(camera movement, cuts, fades). It also disables prompt enhancement, which
otherwise rewrites the prompt and voids `--seed` reproducibility.

```bash
./generate_video.py "a candle flame flickers in a dark room" \
    --image frame.png --loop --duration 4 --resolution 720p --seed 42 --no-audio
```

What makes the seam hold up:

- Prompt **cyclical motion**, not a narrative arc — rippling water, drifting
  steam, an object completing one full rotation. Anything that ends somewhere
  different from where it started will fight the constraint.
- Keep it **short** (`--duration 4`); there is less room to wander.
- Pass **`--no-audio`**; the soundtrack does not loop cleanly even when the
  picture does.
- Supply your own `--negative-prompt` to override the built-in one if your
  scene needs different suppressions — `--loop` only fills it in when you
  haven't.

Longer loops are not directly supported. The API's video-extension feature
(+7 s per call) is the other lever, and is not wired into this script.

## Constraints the API enforces

These are checked locally, before any request is sent:

| Rule | |
| --- | --- |
| `--last-frame` | Only valid alongside `--image` |
| `--reference-image` | Max 3; cannot combine with `--image` |
| `--duration` | Must be `8` at `1080p` or `4k`, and with reference images |

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

**The same prompt and `--seed` gave different videos.** Prompt enhancement
rewrites your prompt before generation and defaults to on, which voids seed
reproducibility. `--loop` turns it off; there is currently no standalone flag
to do so on other paths.

**A generation was blocked.** Veo runs safety filters over the audio as well as
the video and will sometimes block on the audio alone. Blocked generations are
not charged. `--no-audio` avoids that class of failure entirely.

**The video URL 404s a few days later.** Generated videos are deleted from the
server after 2 days. The script downloads immediately, so this only bites if
you are reusing an old operation.
