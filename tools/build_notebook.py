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

**How long:** first run downloads ~10-15 GB of model weights and takes
10-30 minutes. After that, one 5-second clip is roughly 1-3 minutes.

**VRAM reality check.** A free T4 has ~15 GB. That runs **Wan 2.2 TI2V-5B**
and **LTX** comfortably, and quantized 14B models slowly. It will *not* run
full-precision Wan 14B or HunyuanVideo — don't waste hours trying.
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
    "No GPU. In the right-hand panel: Settings -> Accelerator -> GPU T4 x2,"
    " then Session -> Restart."
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
        """REPO_URL = "https://github.com/YOUR_USERNAME/kidvid.git"

import shutil

repo_dir = WORK / "kidvid"
if repo_dir.exists():
    shutil.rmtree(repo_dir)

if "YOUR_USERNAME" in REPO_URL:
    print(
        "Placeholder REPO_URL still set.\\n"
        "Either (a) push this project to your GitHub and paste the URL above,\\n"
        "or (b) upload the kidvid/ folder via the Kaggle file browser."
    )
else:
    !git clone --depth 1 {REPO_URL} {repo_dir}
    print("cloned into", repo_dir)

# ffmpeg does the joining, music muxing and captions.
!apt-get -qq install -y ffmpeg espeak-ng > /dev/null
print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
"""
    ),
    md(
        """## 4. Install WanGP

[WanGP](https://github.com/deepbeepmeep/Wan2GP) is the friendly front end for
the open video models. Its own license lets you keep and sell what you make;
you just can't resell WanGP itself as a hosted service.

We install it into a **separate directory** because it pins its own torch
version, which would otherwise clash with the Kaggle image.
"""
    ),
    code(
        """!pip install -q "huggingface_hub[cli]" einops ftfy sentencepiece omegaconf

wangp_dir = WORK / "Wan2GP"
if not wangp_dir.exists():
    !git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP {wangp_dir}

!ls {wangp_dir}
"""
    ),
    code(
        """# Find out which Python entrypoints this WanGP build actually exposes.
# WanGP's CLI flags move between releases, so we query instead of guessing.
%cd {wangp_dir}
!python {repo_dir}/tools/wangp_bridge.py --list
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
        """PROMPT = "a brave little turtle who learns to share"

# Optional: one image per scene. Leave empty for pure text-to-video.
# Upload via the Kaggle file browser, then list them here.
IMAGE_PATHS: list[str] = []

SCENES = 4
SECONDS_PER_SCENE = 5.0
WIDTH, HEIGHT = 832, 480

# Lower for speed, raise for quality. 30 is a good default for Wan.
STEPS = 30

# 4-8 steps with a "Lightning"/distilled checkpoint - much faster, slightly softer.
# STEPS = 6

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

This is the slow part. Each scene loads the model, samples, and saves an MP4.
Progress prints as it goes, so you can watch the first clip finish before
committing to the rest.
"""
    ),
    code(
        """# Make kidvid importable and run only the video stage, so a failure here
# doesn't throw away music you already made.
import os
os.chdir(WORK)
sys.path.insert(0, str(WORK))

from kidvid.wangp_backend import WanGPBackend

backend = WanGPBackend(model="wan2.2_ti2v_5B")
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
unless you save them. To keep them, point WanGP's checkpoint directory at
`/kaggle/working` instead — but note the 20 GB output limit.

Useful tweaks:

| Want | Change |
| --- | --- |
| Faster | `STEPS = 6` with a Lightning/distilled checkpoint |
| Longer clips | raise `SECONDS_PER_SCENE` (VRAM grows fast) |
| More scenes | raise `SCENES`; the storyboard cycles beats automatically |
| A real voice | swap `EspeakNarrator` in `kidvid/tts.py` for Kokoro or Piper |
| Better writing | pass an LLM into `build_storyboard` to replace the templates |

### Licensing, in one paragraph

Wan 2.2, LTX-Video, ACE-Step and MusicGen are Apache-2.0 or MIT — you can use
what you make, including commercially. HunyuanVideo ships under Tencent's own
license with restrictions, so read it before shipping anything made with it.
WanGP is free to use and you may sell its output, but you may not resell WanGP
itself as a hosted or paid service. Everything you generate here is yours.
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
