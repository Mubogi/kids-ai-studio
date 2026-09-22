# AGENTS.md

Repository notes for future sessions.

## What this is

`kidvid` builds animated kids' story videos and songs from a single text
prompt, using open-source video/music models on a free Kaggle GPU.

## Commands

```bash
python3 tools/smoke_test.py                    # full pipeline, mock backends, needs ffmpeg
python -m kidvid --dry-run "<prompt>"          # storyboard only, no rendering
python tools/build_notebook.py                 # regenerate notebooks/*.ipynb
```

Regenerate the notebook from `tools/build_notebook.py` — never hand-edit the
`.ipynb`. It is generated so it stays reviewable in diffs.

## Architecture conventions

- **Backends are swappable and duck-typed.** `MockVideoBackend` /
  `MockMusicBackend` require no GPU and no weights. Any change must keep the
  mock path working, because it's the only thing testable off-GPU.
- **`storyboard.py` is pure Python with no LLM.** Deterministic by design:
  the same prompt always yields the same storyboard. If you add an LLM, keep
  the template version as the fallback.
- **Heavy model stacks are never added to `requirements.txt`.** They pin
  conflicting torch/CUDA versions and only belong in the Kaggle notebook.
- **WanGP is driven via `tools/wangp_bridge.py`**, which introspects entrypoints
  at runtime rather than hard-coding flags. WanGP's flags change between
  releases; do not replace this with hard-coded CLI args.

## Touched files

| Want | Change |
| --- | --- |
| Rebrand / recolour | `kidvid/theme.py` only — everything reads from that palette |
| Rewrite the stories | Replace `build_storyboard` in `kidvid/storyboard.py` |
| A nicer voice | Replace `EspeakNarrator` in `kidvid/tts.py` (Kokoro, Piper) |

## Branding

This is a **Jordan Design Hub (JD Hub)** project, part of the Mubogi Gastavas
Jordan Tech Ecosystem. The palette is JD Hub yellow (`#FFC107`) in
`kidvid/theme.py`. Scene cards, the title card, song captions and the generated
preview page all read from that one `Theme` dataclass — never hard-code a colour
in `movie.py` or `video.py`.

`PROJECT_PROFILE.md` follows the ecosystem template
(https://github.com/Mubogi/mubogi-ecosystem `PROJECT_TEMPLATE.md`). Keep the
section structure identical; update only content. The same profile belongs in
`profiles/<project>.md` in the ecosystem repo.

## GPU / environment facts (verified 2026-09)

- A free Kaggle T4 has ~15 GB VRAM. It runs Wan 2.2 **TI2V-5B** and **LTX**.
  It cannot run full Wan 14B or HunyuanVideo.
- `huggingface-cli download extra-model` with quotes works around a HF Hub
  version comparison TypeError.
- ffmpeg is required locally for any test; `espeak-ng` only for `--narrate`.

## Licensing constraints

- Wan 2.2, LTX-Video, Open-Sora, ACE-Step: Apache-2.0. MusicGen: MIT.
- **HunyuanVideo is NOT OSI open source** (Tencent custom license). Flag this
  to the user before any commercial use.
- WanGP: output is yours to sell, but reselling WanGP itself as a hosted or
  paid service needs a separate license.
- Official Wan open weights stop at **2.2**. "Wan 2.7 weights" downloads are
  likely malware — warn the user if they mention one.

## Gotchas

- ffmpeg's concat demuxer needs identical streams, so `movie._uniformize()`
  re-encodes every clip before joining. Don't remove it.
- `ffmpeg_color()` is required for drawtext colours: ffmpeg wants `0xRRGGBB`
  (and `@alpha`), not `#RRGGBB`. Passing a raw CSS hex silently renders wrong.
- Parent directories must exist before `file_editor` create; `mkdir -p` first.
- The shell here rejects multiple top-level commands; chain with `&&`.

## Publishing (state as of 2026-09-22)

- **GitHub: live.** Pushed to https://github.com/Mubogi/kids-ai-studio
  (`master` = initial import, `mubogi-branding` = PR #1).
  The ecosystem portfolio PR is Mubogi/mubogi-ecosystem#1.
- **Kaggle: published**, at https://www.kaggle.com/code/jordangastavas/kids-ai-studio
  (account `jordangastavas`, notebook private). But it **cannot generate
  video yet** - see the accelerator blocker below.
- There is **no Kaggle "connect a repo"** feature. The notebook clones the
  repo at runtime, so GitHub stays the single source of truth.
- Never store a token in `.git/config`; pass it inline per command instead.
  Push with `https://$TOKEN@github.com/...` and it stays out of the config.
- The default `GITHUB_TOKEN` in this environment is a *different, read-only*
  account (`jun123432`). Pushes to Mubogi need an explicit Mubogi token.

### Kaggle accelerator blocker (verified, not a guess)

Requesting `enable_gpu` and `enable_internet` in `kernel-metadata.json`
**succeeds at the API level** - the stored metadata really does say
`machine_shape: NvidiaTeslaT4` and `enable_internet: true` - but the running
kernel gets **neither**. A probe notebook printed:

    cuda available: False
    internet error: URLError Temporary failure in name resolution

GPU quota is not the cause: `kaggle quota` reports 30h remaining, 0h used.
Kaggle gates GPU, TPU and internet behind **account verification**, and the
CLI's own 403 help text points at it ("Your account is missing phone or
identity verification. Verify at https://www.kaggle.com/settings").

So: **verify the Kaggle account first** (phone number, plus identity
verification for some features). Until then the notebook stops at step 1 by
design, with an assertion that says exactly that.

### Kaggle API gotchas learned the hard way

- **Public notebooks cannot be created over the API.** `is_private: false`
  returns `403 Forbidden` on `SaveKernel`. Bisected with two identical
  minimal notebooks: private works, public 403s. Dataset writes are fine
  either way, so this is specific to kernels, not a token scope problem.
  `publish_kaggle.py` therefore defaults to private; add `--public` to opt
  in, or flip visibility in the notebook's Settings on the web UI.
- **`nvidia-smi` is not on PATH** in the Kaggle Python image. Never call it
  unguarded; use `shutil.which` and rely on `torch.cuda.is_available()`.
- Pushing a notebook **auto-triggers a run**. Poll with
  `python3 -m kaggle kernels status <user>/<slug>` and read failures with
  `python3 -m kaggle kernels output <user>/<slug> -p <dir>`.
- The `kaggle` console script installs to `~/.local/bin`, which is not on
  PATH here. Use `python3 -m kaggle ...` or the resolver in
  `tools/publish_kaggle.py`.

## GitHub push is blocked (expired token)

The `GITHUB_TOKEN` in this environment returns **401 Bad credentials** from
the GitHub API (`/user`), so pushes to `github.com/Mubogi/kids-ai-studio`
cannot be authenticated. The commit is ready on branch `mubogi-branding`;
it needs a fresh token (repo scope) to push. Nothing else about the build
depends on this.

## Caption rendering: three ffmpeg/libass traps

All three were found by testing, and each one fails *silently* - captions
just do not appear, with no error from ffmpeg.

1. **SRT is laid out on a fixed 288-unit-tall canvas**, not on the real
   frame size. Style values must be scaled by `288 / frame_height`, or a
   1080p caption ends up 6.7x too large and off-screen.
2. **`BorderStyle=3` with `Outline=0` renders nothing at all.** The box
   style needs `Outline>=1` (1 is the minimum padding).
3. **ASS centiseconds must be two digits** (`0:00:04.90`, not
   `0:00:04.9`). Single digits make libass drop the whole file. Note the
   dedicated `ass` filter cannot parse ASS in this ffmpeg build either,
   so `kidvid/social.py` burns an SRT via `force_style` instead.

## The studio interface

`python3 -m kidvid.serve` starts the FastAPI app on port 12000 (exposed at
the work-1 preview URL). One background worker processes jobs one at a
time; the pipeline stdout is captured and parsed into progress events.
Front end is a single static file, `kidvid/web/index.html`.

## ROOT CAUSE FOUND: Kaggle gives this account no GPU

A minimal diagnostic notebook (`jordangastavas/gpu-check`) settled it:

    nvidia-smi on PATH: False
    torch: 2.10.0+cpu
    cuda available: False
    device count: 0
    AssertionError: NO GPU GRANTED to this account

The notebook metadata correctly requested `enable_gpu: true` and
`accelerator: nvidiaTeslaT4`. Kaggle honoured neither. **Kaggle only grants
GPU/TPU to accounts with a verified phone number**, so without verification
every run silently falls back to CPU.

Note the CPU torch build: `2.10.0+cpu`. A GPU session gets a CUDA build, so
this is proof of a CPU-only session, not a driver problem.

**Fix (user action, cannot be automated):** kaggle.com -> Settings ->
Phone Verification -> enter SMS code -> reopen notebook -> Settings ->
Accelerator -> GPU T4 x2 -> Session -> Restart. Some carriers and most VoIP
numbers reportedly fail Kaggle's SMS check.

## T4 cannot run the bf16 Wan 2.2 checkpoint

The notebook originally defaulted to `wan2.2_ti2v_5B` in bf16, which needs
~24GB. Kaggle's free GPU is a T4: 16GB and **sm_75, which has no bfloat16
support**. It would have failed on a GPU session too. The default is now
`wan2.2_ti2v_5B_Q4_K_M`, a GGUF quantisation that offloads to system RAM and
fits in ~8GB. On a 24GB+ card, pass `wan2.2_ti2v_5B` for better quality.

Model choice guidance (2026):
- Wan 2.2 TI2V-5B is the best fit for a free T4 - 720p, Apache-2.0,
  text-to-video *and* image-to-video.
- HunyuanVideo 1.5 (14GB fp8) is the quality leader but needs more VRAM
  headroom and is slow on a T4.
- LTX-2.3 is the only open model with native synced audio, but wants 16GB
  at Q4 and a newer card for the best precision.

## Realistic generation speed on a T4

A 5-second 480p clip at 30 steps is roughly 2-6 minutes. A 4-scene
20-second story is roughly 10-25 minutes. The first scene is slowest
because the model loads then. `STEPS=6` plus a Lightning checkpoint is the
main speed lever.

## Pollinations went paid - Stable Horde is the new free image source

Discovered while testing the new style presets. Anonymous Pollinations
requests now fail hard:

    {"error":"Internal Server Error","message":"Gen Sana request failed with
     402: ... Insufficient balance. This request costs ~0.0001 pollen, but
     your available balance is 0.0000."}

and the queue endpoint returns 429 with `maxAllowed: 1` per IP. The free
keyless image tier that `ImageVideoBackend` was built on is gone, so the
whole `--video-backend ai` path was silently dead until now.

`HordeImageBackend` replaces it, verified end to end:

  * POST /api/v2/generate/async with apikey `0000000000` (the documented
    anonymous key - no signup)
  * poll /generate/check, then read /generate/status
  * typical anonymous wait ~75s for a 1024x576 still

Two traps worth remembering:

1. The signed URL in `generations[0].img` must be used **in full**. It
   carries an X-Amz-Signature query string; truncating it (or logging only
   a prefix, which is how I first hit this) yields
   `400 InvalidArgument Authorization` from Cloudflare R2.
2. `is_possible: false` in the check response means no worker can ever
   serve the request. Fail fast instead of waiting out the timeout.

Set `HORDE_API_KEY` to a free personal key to skip most of the queue.
Set `POLLINATIONS_TOKEN` to use Pollinations instead.

## Art styles

"Real moving people" was impossible before for a reason that had nothing to
do with GPUs: `SAFE_STYLE` hard-coded "colorful 2D animated cartoon" into
every prompt, and `NEGATIVE_PROMPT` actively banned "realistic human faces".
The model was told to avoid exactly what was wanted.

`storyboard.STYLE_PRESETS` now offers cartoon / cinematic / 3d / anime, and
each has a matching negative in `NEGATIVE_PRESETS` so the two cannot
disagree. Selectable in the UI as "Look". Default output is now 1024x576
(was 832x480) - the largest frame the services render.

Note the honest limitation that remains: these are AI *images* with a
camera move, not diffusion video. Real frame-by-frame motion still needs the
WanGP path and a GPU. See the GPU section above.
