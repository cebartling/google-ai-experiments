# Veo text-to-video

`generate_video.py` drives Veo through the Gemini API. It is a single
[uv](https://docs.astral.sh/uv/) script — dependencies are declared inline, so
there is no venv to create.

## Setup

```bash
cp .env.example .env.local   # then fill in GEMINI_API_KEY
chmod +x generate_video.py
```

The key comes from <https://aistudio.google.com/apikey>. Every generation is
billed, so use `--dry-run` while you are still working out the flags.

## Usage

```bash
./generate_video.py "A drone flies through a canyon at sunset"
./generate_video.py --prompt-file prompt.txt --output canyon.mp4
```

Run the tests with `uv run test_generate_video.py`.

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
