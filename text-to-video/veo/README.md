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
- `ffmpeg` (with `ffprobe`) on `PATH` — only for `--crossfade`; nothing
  else in the script needs it.

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
the flags. `--dry-run` also prints what the request would cost.

### The quota guards

The Gemini API enforces a **spend-based rate limit over a rolling 10-minute
window** — $10 on Tier 1 — and returns a bare `429 RESOURCE_EXHAUSTED` when you
cross it, with no `RetryInfo` and no endpoint to ask how much headroom is left.
At $0.40/second for Veo 3.1 Standard, three 8-second clips ($9.60) is enough to
trip it, which is easy to do: a seamless loop takes two generations.

There is a second, blunter limit too: **10 Veo requests per day** on Tier 1,
resetting at midnight Pacific. In practice this is the one that bites — ten
8-second clips is an ordinary afternoon, and a seamless loop costs two of them.

So the script keeps its own books. A request's cost is deterministic before it
is sent — model x resolution x duration — so each request is appended to
`.spend_ledger.json` (gitignored, beside the script), and the next one is
refused if it would cross either limit:

```
Error: 10 of 10 requests used today; the daily quota resets at midnight
Pacific, in 4h18m. Pass --force to send it anyway.

Error: this request costs $3.20; $9.60 already spent in the last 10 minutes
against a $10.00 limit. Wait 6m12s, or pass --force to send it anyway.
```

The daily quota is checked first, since it is the one that cannot be waited out
in minutes.

Worth knowing:

- **Both caps assume Tier 1**, since the API will not say which tier a key is
  on: $10.00 and 10 requests/day. Override either in `.env.local` with
  `VEO_SPEND_LIMIT_USD` ($50 on Tier 2, $200 on Tier 3) or
  `VEO_DAILY_REQUEST_LIMIT`. Your real numbers are on the
  [AI Studio rate-limit page](https://aistudio.google.com/rate-limit), which is
  the only place they are visible — the API exposes no way to query them.
- **A model with no published price is never blocked.** `--model` takes an
  arbitrary string; an unrecognised one prints a note and the request goes out.
- **A model with no published price still counts against the daily quota.**
  Only the spend check needs a price; the request count does not.
- **The ledger is advisory, not authoritative.** It records what this script
  did — generations from AI Studio or another machine are invisible to it, and
  entries are written when a request is *accepted*, so one that Veo's safety
  filters block afterwards is counted but not charged. Entries are kept until
  the next Pacific midnight, so the daily count has the whole day to work with.
- **A corrupt or unwritable ledger never blocks a run.** Delete it to reset:
  `rm .spend_ledger.json`.

## Usage

```bash
./generate_video.py "A drone flies through a canyon at sunset"
./generate_video.py --prompt-file prompt.txt --output canyon.mp4
```

Run the tests with `uv run test_generate_video.py`.

## How a run proceeds

Video generation is a long-running operation, so a run is three phases and
takes minutes, not seconds:

1. **Kick off.** The request returns an operation handle immediately. HTTP 5xx
   is retried up to five times with exponential backoff. A 429 rate limit gets
   only **two** attempts, and none at all when the error names a per-day quota
   — see below. Anything else (a bad key, a rejected argument) fails straight
   away rather than being retried.
2. **Poll.** The script checks the operation every `--poll-seconds` and prints
   elapsed time to stderr, giving up at `--timeout-seconds`. Google documents
   latency of 11 seconds to 6 minutes, so the 600 s default has room but a
   generation that hits the ceiling during peak hours will need it raised.
3. **Download.** The finished video is fetched and written to `--output`, which
   is **overwritten without warning** if it already exists.

Progress goes to stderr and the saved path to stdout, so `--output` paths can be
piped somewhere useful.

### Why 429s get a shorter leash

A 429 is two different failures wearing the same status code. A per-minute
limit clears in seconds and is worth retrying. A per-day quota does not clear
until midnight Pacific, and **every retry spends another request from the quota
that is already exhausted** — costly when Veo allows only 10 requests a day on
Tier 1, where five attempts can burn half a day's budget chasing a limit no
amount of waiting will lift.

The API does not reliably distinguish them. Google *sometimes* includes a
`QuotaFailure` detail naming the violated quota, and when it names a per-day
one the script does not retry at all. When the detail is absent — as it often
is — the script falls back to two attempts rather than five: enough to ride out
a per-minute limit, cheap enough to be wrong about.

## Flags

| Flag | Purpose |
| --- | --- |
| `prompt` / `--prompt-file` | The prompt, inline or from a file |
| `--negative-prompt` | What should *not* appear |
| `--image` | Still to use as the first frame |
| `--last-frame` | Still to end on; requires `--image` |
| `--reference-image PATH:TYPE` | Up to three refs, `asset` or `style` |
| `--loop` | End on the starting frame; requires `--image` |
| `--crossfade [SECONDS]` | Dissolve the tail over the head to hide the seam (default 0.5); requires `--loop` |
| `--duration {4,6,8}` | Clip length in seconds (default 8) |
| `--person-generation` | `dont_allow`, `allow_adult`, `allow_all` |
| `--aspect-ratio`, `--resolution` | `16:9`/`9:16`, `720p`/`1080p`/`4k` |
| `--dry-run` | Print the resolved request and exit; no API call |
| `--force` | Send the request even if it would cross the spend limit or daily quota |
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

#### Tightening the seam with `--crossfade`

`--loop` alone usually leaves a visible hitch: Veo's final frame drifts from
the anchor even with the negative prompt suppressing camera motion — typically
the framing pulls slightly wider. `--crossfade` closes that gap after the
download by dissolving the clip's last half-second over its first half-second,
so the wrap is a blend rather than a cut.

```bash
./generate_video.py "a candle flame flickers in a dark room" \
    --image frame.png --loop --crossfade --duration 8 --resolution 720p
```

Pass a value to change the fade length: `--crossfade 0.75`. Round trips
cleanly when the fade is a whole number of frames — Veo returns 24 fps, so
0.5 s is 12 frames.

Two things to know:

- **The clip gets shorter by the fade length.** An 8-second generation with
  `--crossfade 0.5` lands as a 7.5-second file. That is inherent: the tail is
  consumed by the dissolve rather than played.
- **The audio is faded to match** (`afade` in/out summed at `normalize=0`, so
  the level does not dip through the blend), which softens — but does not
  remove — the audio seam noted above. Strip it if it still distracts.

Measured on one 8-second clip, the seam's mean luma difference went from 32.5
to 14.4, against an adjacent-frame baseline of 23.4 — i.e. from a step larger
than ordinary motion to one smaller than it. The clip re-encodes at CRF 16,
and the original is only replaced once ffmpeg succeeds.

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
