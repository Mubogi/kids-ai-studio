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

- **Nothing is pushed anywhere yet.** The `GITHUB_TOKEN` in this environment
  authenticates as `jun123432` and is *read-only* for `Mubogi` — verified by a
  failed ref-creation API call returning "Resource not accessible by
  integration". Do not claim a push succeeded without re-testing writes.
- There is **no Kaggle "connect a repo"** feature. `tools/publish_kaggle.py`
  pushes the notebook via the Kaggle API; the notebook then clones the repo at
  runtime. GitHub stays the single source of truth.
- Kaggle API tokens live at `~/.kaggle/kaggle.json` and are not present in this
  environment.
- Never store a token in `.git/config`; pass it inline per command instead.
