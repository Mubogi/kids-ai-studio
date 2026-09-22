#!/usr/bin/env python3
"""Build notebooks/kidvid_kaggle.ipynb.

Generating the .ipynb from code keeps the notebook reviewable and diffable
instead of being an unreadable blob of JSON in git history.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "kidvid_kaggle.ipynb"

# nbformat >= 5 requires a stable, unique id on every cell. Kaggle's validator
# warns loudly without them.
_id_counter = 0


def _next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md-{_next_id()}",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"code-{_next_id()}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md(
        """# Make kids' story videos and songs on a free Kaggle GPU

Turns one prompt into an animated cartoon — or a singable song — with pictures,
music, and narration.

**Before you run:** `Settings → Accelerator → GPU T4 x2` and `Internet → On`.
Without the GPU accelerator this notebook will stop at step 1.

**How long:** first run downloads ~19 GB of model weights and takes
10-30 minutes. After that, one 5-second 480p clip is roughly 2-6 minutes
with offloading enabled.

**VRAM reality check.** A free T4 has ~15.6 GB and is **sm_75, so it has no
bfloat16** — fp16 is the only usable half precision there. Wan 2.2 TI2V-5B
is the right model for this card: the 5B transformer (~10 GB) plus its
umt5-xxl text encoder (~9 GB) exceed the card together, so the bridge runs
with model-level CPU offload. It will *not* run the 14B Wan models or
HunyuanVideo — don't waste hours trying. A 24GB+ card can turn offloading
off for a large speedup.
"""
    ),
    md("## 1. Check the GPU"),
    code(
        """import shutil, subprocess

# nvidia-smi is not always on PATH in Kaggle images, so treat it as optional.
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi not on PATH; falling back to torch")

import torch
assert torch.cuda.is_available(), (
    "No GPU was granted to this Kaggle account.\n\n"
    "Setting the accelerator is not enough on its own. Kaggle only hands out\n"
    "GPU/TPU capacity to accounts with a VERIFIED PHONE NUMBER:\n"
    "  1. kaggle.com -> your avatar -> Settings -> Phone Verification\n"
    "  2. enter the SMS code\n"
    "  3. reopen the notebook -> Settings -> Accelerator -> GPU T4 x2\n"
    "  4. Session -> Restart\n\n"
    "Until then every run silently falls back to CPU (the torch build is\n"
    "+cpu), which cannot generate video at all."
)
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
"""
    ),
    md(
        """## 2. Where things live

Kaggle gives you `/kaggle/working` (~20 GB, kept when you save) and
`/kaggle/temp` (scratch). Model weights are large, so they go in
`/kaggle/temp` unless you want them in the saved output.
"""
    ),
    code(
        """from pathlib import Path

WORK = Path("/kaggle/working")
MODELS = Path("/kaggle/temp/models")

for d in (WORK, MODELS):
    d.mkdir(parents=True, exist_ok=True)

print("working:", WORK)
print("models: ", MODELS)
print(subprocess.run(["df", "-h", "/kaggle"], capture_output=True, text=True).stdout)
"""
    ),
    md(
        """## 3. Get the code

Set `REPO_URL` to your own fork once you push this project, or leave the
placeholder and paste the files in manually. Nothing here needs a private repo.
"""
    ),
    code(
        """# Points at the real project. If you forked it, change the URL; if you
# uploaded kidvid/ by hand instead, set REPO_URL = "" to skip the clone.
REPO_URL = "https://github.com/Mubogi/kids-ai-studio.git"
# Empty means "whatever the repo's default branch is", which is where the
# working code lives. Only pin this if you need a specific branch.
REPO_BRANCH = ""

import shutil

repo_dir = WORK / "kidvid"
if repo_dir.exists():
    shutil.rmtree(repo_dir)

_branch = f"--branch {REPO_BRANCH} " if REPO_BRANCH else ""
if not REPO_URL:
    print(
        "REPO_URL is empty.\\n"
        "Either (a) set it to your own fork, or\\n"
        "(b) upload the kidvid/ folder via the Kaggle file browser."
    )
else:
    !git clone --depth 1 {_branch}{REPO_URL} {repo_dir}
    print("cloned into", repo_dir)

# ffmpeg does the joining, music muxing and captions.
!apt-get -qq install -y ffmpeg espeak-ng > /dev/null
print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
"""
    ),
    md(
        """## 4. Install the video model runtime

The real motion comes from **Wan 2.2 TI2V-5B** driven through `diffusers`.
It is Apache-2.0 licensed, does text-to-video *and* image-to-video, and is
small enough to fit a T4 once offloading is on.

Its output is yours to keep and sell; there is no hosted-service restriction
on the weights, unlike some of the web UIs.

We deliberately do **not** install a second copy of torch. Kaggle's image
already ships a CUDA build (`2.10.0+cu128`), and reinstalling torch is the
most common way to break a Kaggle GPU session.
"""
    ),
    code(
        """!pip install -q "diffusers>=0.35" transformers accelerate safetensors \\
    imageio imageio-ffmpeg ftfy sentencepiece
print("install done")
"""
    ),
    code(
        """import torch, diffusers
print("torch:", torch.__version__)
print("diffusers:", diffusers.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    # A T4 is sm_75 and has no bfloat16. Printed up front because it decides
    # which dtype the bridge may use.
    print("bfloat16 supported:", torch.cuda.is_bf16_supported())
"""
    ),
    md(
        """## 5. Write your prompt

Three shapes work best:

- **Story:** `a brave little turtle who learns to share`
- **Song:** `a song about a friendly dragon`
- **From your pictures:** `my dog goes to space` plus `IMAGE_PATHS` below
"""
    ),
    code(
        """import os

PROMPT = "a brave little turtle who learns to share"

# Optional: one image per scene. Leave empty for pure text-to-video.
# Upload via the Kaggle file browser, then list them here.
IMAGE_PATHS: list[str] = []

SCENES = 4
# Short scenes on purpose. Generation cost is roughly linear in frames, and
# the free GPU quota cannot absorb many 5-second 30-step clips per week -
# see the timing note further down for the measured numbers.
SECONDS_PER_SCENE = 3.0
WIDTH, HEIGHT = 832, 480

# Lower for speed, raise for quality. Wan's usual default is 30; 12 keeps a
# story inside the GPU quota, and goes softer rather than breaking.
STEPS = 12

# VRAM left free on each card so cuBLAS can allocate its GEMM workspace.
# Without this the run dies with CUBLAS_STATUS_ALLOC_FAILED partway through.
# Raise it if you still hit that, lower it if the model will not fit.
os.environ.setdefault("WAN_MAX_MEM_HEADROOM_GB", "2")

# Uncomment to add a robot-but-clear narrator (free, offline).
# NARRATE = True
NARRATE = False
"""
    ),
    md("### Preview the storyboard before spending GPU time"),
    code(
        """import sys
sys.path.insert(0, str(WORK))

from kidvid.config import ShowConfig

cfg = ShowConfig(
    prompt=PROMPT,
    title=PROMPT.title()[:34],
    scene_count=SCENES,
    seconds_per_scene=SECONDS_PER_SCENE,
    width=WIDTH,
    height=HEIGHT,
    steps=STEPS,
    image_paths=IMAGE_PATHS,
    narrate=NARRATE,
    make_music=True,
    out_dir=str(WORK / "output"),
)
cfg.music_seconds = SCENES * SECONDS_PER_SCENE

from kidvid.storyboard import build_storyboard

board = build_storyboard(cfg)
print(f"character: {board.character}")
print(f"music:     {board.music_prompt}\\n")
for s in board.scenes:
    kind = "image-to-video" if s.image else "text-to-video"
    print(f"--- scene {s.index + 1} [{s.beat}] {kind} ---")
    print(s.video_prompt[:160] + "...")
if board.song:
    print("\\n--- song ---")
    for line in board.song:
        print("  " + line)
"""
    ),
    md(
        """## 6. Generate the clips

This is the slow part. Measured on a T4 x2 with balanced placement, a
1-second 480p clip at 6 steps takes about **4-6 minutes**, so budget roughly
**20-30 seconds per frame of output**. Wan is a diffusion video model: it is
not real-time, and this is the honest cost of real motion rather than a
pan-and-zoom over a still.

The GPU quota is what makes that tight - roughly 9-12 hours of T4 per week
on a free account, and one 20-second story can be most of an hour. Two
levers:

- lower `SCENES` and `SECONDS_PER_SCENE` first; they cut cost linearly
- `STEPS` below 30 trades detail for speed, and gets noticeably softer

The model loads once per run, not per scene, so the first clip is the
slowest. Progress prints as it goes, so you can watch scene 1 finish before
committing to the rest.
"""
    ),
    code(
        """# Make kidvid importable and run only the video stage, so a failure here
# doesn't throw away music you already made.
import os
os.chdir(WORK)
sys.path.insert(0, str(WORK))

from kidvid.wangp_backend import DEFAULT_WAN_MODEL, WanGPBackend

# fp16 plus model-level CPU offload: the 5B transformer (~10GB) and the
# umt5-xxl text encoder (~9GB) total ~19GB, which does not fit a T4's 15.6GB
# together. Offload keeps one component on the GPU at a time.
#
# On a 24GB+ card (A100, L4, 4090) pass offload="none" for a large speedup,
# and dtype="bf16" if the card supports it.
backend = WanGPBackend(model=DEFAULT_WAN_MODEL, dtype="fp16", offload="auto")

clips_dir = Path(cfg.out_dir) / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)

clips = []
for scene in board.scenes:
    target = clips_dir / f"scene_{scene.index:02d}.mp4"
    if target.exists():
        print(f"scene {scene.index + 1}: already done, skipping")
        clips.append(target)
        continue
    print(f"scene {scene.index + 1}/{len(board.scenes)} ...")
    backend.generate(scene, cfg, target)
    clips.append(target)

print("\\nclips:", [c.name for c in clips])
"""
    ),
    md(
        """## 7. Music

ACE-Step (Apache-2.0) is fast and song-like. MusicGen is a reliable fallback.
If neither installed cleanly, the mock backend still produces a simple
melody so you get a finished video to look at.
"""
    ),
    code(
        """!pip install -q acestep 2>/dev/null || echo "ACE-Step unavailable, will fall back"
"""
    ),
    code(
        """from kidvid.music import default_backend, MockMusicBackend

music_path = Path(cfg.out_dir) / "music.m4a"
try:
    mbackend = default_backend()
    print("music backend:", type(mbackend).__name__)
    mbackend.generate(board.music_prompt, cfg.music_seconds, music_path)
except Exception as exc:
    print(f"real music backend failed ({exc}); using the mock melody")
    MockMusicBackend().generate(board.music_prompt, cfg.music_seconds, music_path)

print("music:", music_path, music_path.stat().st_size // 1024, "KB")
"""
    ),
    md("## 8. Assemble the movie"),
    code(
        """from kidvid.movie import concat_clips, mux, add_song_captions, burn_title
from kidvid.util import probe_duration

out_dir = Path(cfg.out_dir)
silent = concat_clips(clips, cfg, out_dir / "video_silent.mp4")
print(f"joined: {probe_duration(silent):.1f}s")

finished = mux(silent, music_path if music_path.exists() else None,
               out_dir / "final.mp4", cfg)

if board.song:
    finished = add_song_captions(finished, board.song, 0.0,
                                 out_dir / "final_captioned.mp4", cfg)
else:
    finished = burn_title(finished, cfg.title, out_dir / "final_titled.mp4", cfg)

print("FINAL:", finished)
print("duration:", round(probe_duration(finished), 1), "s")
print("size:", round(finished.stat().st_size / 1e6, 1), "MB")
"""
    ),
    md(
        """## 9. Watch it and download

The inline player works for short clips. For anything longer, download the file
from the **Output** panel on the right — Kaggle keeps `/kaggle/working` when you
save the version.

This also builds a **branded preview page**: a single standalone HTML file with
the video, storyboard, lyrics and the JD Hub yellow theme. Download it and open
it in any browser — no server, no internet needed.
"""
    ),
    code(
        """from IPython.display import Video, display

display(Video(str(finished), embed=True, width=640))

import shutil
target = WORK / "my_kids_video.mp4"
shutil.copyfile(finished, target)

# Branded standalone preview page (video + storyboard + lyrics, JD Hub yellow).
# Only present once the branding branch is merged, so don't fail the run on it.
import os, subprocess
preview_script = Path(repo_dir) / "tools" / "build_preview.py"
if preview_script.exists():
    subprocess.run(["python", str(preview_script), str(target),
                    str(Path(cfg.out_dir) / "storyboard.json")], check=False)
else:
    print("preview page skipped: tools/build_preview.py not in this revision")
    print("  merge the branding PR, or pull the branch, then re-run this cell")
    if board.song:
        print("  (lyrics for reference)")
        for line in board.song:
            print("   ", line)

print("\ndownload from the Output panel:")
for f in sorted(Path(WORK).glob("*.mp4")) + sorted(Path(WORK).glob("*.html")):
    print("  -", f.name, f"({f.stat().st_size / 1e6:.1f} MB)")
"""
    ),
    md(
        """## 10. Coming back later

Kaggle wipes `/kaggle/temp` between sessions, so model weights re-download
unless you save them. To keep them, set `HF_HOME` to a folder under
`/kaggle/working` instead — but note the 20 GB output limit, and the 5B
model plus its text encoder is already most of that.

Useful tweaks:

| Want | Change |
| --- | --- |
| Faster | drop `STEPS` to 15-20, or shorten `SECONDS_PER_SCENE` |
| Longer clips | raise `SECONDS_PER_SCENE`. Keep the frame count at 4n+1; the bridge snaps it for you |
| More scenes | raise `SCENES`; the storyboard cycles beats automatically |
| A real voice | swap `EspeakNarrator` in `kidvid/tts.py` for Kokoro or Piper |
| Better writing | pass an LLM into `build_storyboard` to replace the templates |
| Higher quality | a 24GB+ card with `offload=False` and `dtype="bf16"` |

### Licensing, in one paragraph

Wan 2.2 and ACE-Step are Apache-2.0, MusicGen is MIT — you can use what you
make, including commercially. HunyuanVideo ships under Tencent's own license
with restrictions, so read it before shipping anything made with it.
Everything you generate here is yours.
"""
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
