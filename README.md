# kidvid — Kids AI Studio

> A [Jordan Design Hub (JD Hub)](PROJECT_PROFILE.md) project — Mubogi Gastavas Jordan Tech Ecosystem.

Turn one sentence into a finished animated kids' story video — or a song —
using open-source models on a **free Kaggle GPU**.

```
"a brave little turtle who learns to share"  ->  4 scenes + music + narration -> MP4
"a song about a friendly dragon"             ->  sung lyrics + captions       -> MP4
"my dog goes to space" + your own pictures   ->  image-to-video animation     -> MP4
```

Everything is branded in **JD Hub yellow** (`#FFC107`): scene cards, title card,
song captions, and the preview page. One palette lives in `kidvid/theme.py` —
change it there and the whole look follows.

## What you actually need

| | |
| --- | --- |
| GPU | Kaggle notebook with `GPU T4 x2` (~15 GB VRAM), free, ~30 h/week |
| Disk | 15–20 GB for model weights |
| Money | $0 |
| Local machine | Any — the heavy lifting happens on Kaggle |

## Quick start

**1. Run it locally first.** This works with no GPU and no model downloads,
so you can confirm the whole pipeline before spending Kaggle hours:

```bash
pip install -r requirements.txt
sudo apt-get install -y ffmpeg espeak-ng   # espeak-ng only for --narrate

python -m kidvid --dry-run "a brave little turtle who learns to share"
python -m kidvid --video-backend mock --music-backend mock --narrate \
    "a song about a friendly dragon"
```

That produces `output/final_captioned.mp4` — real independent clips, real music,
real narration. Only the *pictures* are placeholder gradients.

**2. Do it for real on Kaggle.** Push this repo to GitHub, open
[`notebooks/kidvid_kaggle.ipynb`](notebooks/kidvid_kaggle.ipynb) in Kaggle,
set `Settings → Accelerator → GPU T4 x2`, `Internet → On`, put your repo URL in
the `REPO_URL` cell, and run the cells top to bottom.

Or publish it with the API in one command:

```bash
# needs ~/.kaggle/kaggle.json from kaggle.com -> Settings -> API -> Create New Token
python3 tools/publish_kaggle.py \
    --username YOUR_KAGGLE_USER \
    --repo-url https://github.com/YOUR_USER/REPO
```

Kaggle has no "connect a repo" button, so this pushes the notebook into your
Kaggle account with the GPU and internet flags pre-set. The notebook then clones
this repo itself for the library code, which keeps the two in sync — edit the
library, push to GitHub, and the next Kaggle run picks it up.

## Which models, and why

| Model | License | Role | Fits a T4? |
| --- | --- | --- | --- |
| [Wan 2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2) | Apache-2.0 | Text *and* image to video | Yes |
| [Wan 2.2 14B](https://github.com/Wan-Video/Wan2.2) | Apache-2.0 | Higher quality | Quantized, slowly |
| [LTX-Video](https://github.com/Lightricks/LTX-Video) | Apache-2.0 | Fastest, lowest VRAM (8 GB) | Yes, easily |
| [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) | **Tencent custom** | High quality | No |
| [ACE-Step](https://github.com/ace-step/ACE-Step) | Apache-2.0 | Music / songs | Yes |
| [MusicGen](https://github.com/facebookresearch/audiocraft) | MIT | Music fallback | Yes |
| [WanGP](https://github.com/deepbeepmeep/Wan2GP) | WanGP Community License | Driver / UI for the above | Yes |

**Wan 2.2 + WanGP is the recommended path.** Apache-2.0 weights, low-VRAM
engineering already done for you, and one prompt box that accepts both text and
images.

### Licensing

Wan 2.2, LTX-Video, ACE-Step and MusicGen are permissive — you own and may sell
what you generate. **HunyuanVideo is not OSI open source**; it carries Tencent's
own license with restrictions, so read it before commercial use. WanGP lets you
sell its output (credit it if you sell directly) but forbids reselling WanGP
itself as a hosted or paid service.

> Ignore "Wan 2.7 open weights" download pages. Official Wan open weights stop
> at **2.2**. Treat any "2.7 weights" download as a likely malware vector and
> stick to the official [Wan-Video](https://github.com/Wan-Video/Wan2.2) and
> [Wan-AI](https://huggingface.co/Wan-AI) orgs.

## How it works

```
prompt ──> storyboard ──> one clip per scene ──> concat ──> mux ──> MP4
             (4 beats)      (text- or image-     (ffmpeg)   (+ music,
                            to-video)                        + narration)
```

- `kidvid/storyboard.py` — turns a sentence into a 4-beat picture-book arc
  (setup → journey → problem → resolution), kid-safe style tokens, and lyrics.
  Pure Python, no LLM, fully deterministic.
- `kidvid/video.py` — `MockVideoBackend` (gradients, no GPU).
- `kidvid/wangp_backend.py` — real models, via `tools/wangp_bridge.py`.
- `kidvid/music.py` — ACE-Step → MusicGen → mock melody, first available wins.
- `kidvid/movie.py` — ffmpeg joining, ducking music under narration, captions.

### Swapping parts

| Want | Change |
| --- | --- |
| An LLM writing the story | Replace `build_storyboard` in `kidvid/storyboard.py` |
| A nicer voice | Replace `EspeakNarrator` in `kidvid/tts.py` (Kokoro, Piper) |
| Longer / higher-res clips | Raise `seconds_per_scene` / `width` — VRAM grows fast |
| Faster generations | `--steps 6` with a Lightning/distilled checkpoint |

## Files

```
kidvid/           the library
  config.py       ShowConfig — every knob in one dataclass
  storyboard.py   prompt -> beats, prompts, lyrics
  video.py        video backends (mock now, models on GPU)
  wangp_backend.py  adapter for real models
  music.py        music backends
  tts.py          offline narration
  movie.py        ffmpeg assembly
  pipeline.py     end-to-end orchestration
  __main__.py     CLI
notebooks/        Kaggle notebook (generated by tools/build_notebook.py)
tools/            wangp_bridge.py, build_notebook.py, smoke_test.py
examples/         sample input frames
```

## Limits, honestly

- Free Kaggle GPUs are shared and capped (~30 h/week); long jobs can be
  interrupted.
- A T4 will not run full-precision Wan 14B or HunyuanVideo. Don't fight it.
- The template storyboard is competent but formulaic. It's a scaffold — the
  interesting upgrade is dropping an LLM into `build_storyboard`.
- Music and video are generated separately, so they won't be beat-synced.

## Testing

```bash
python3 tools/smoke_test.py     # all three prompt styles, end to end
```

This exercises the real ffmpeg pipeline with the mock backends and asserts the
output duration, so it catches assembly regressions without a GPU.
