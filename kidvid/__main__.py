"""Command line entry point.

    python -m kidvid "a brave little turtle who learns to share"
    python -m kidvid --backend mock --narrate "a song about a friendly dragon"
    python -m kidvid --image photo.png "my dog goes to space"
"""

from __future__ import annotations

import argparse
import sys

from .config import ShowConfig
from .pipeline import make_video
from .storyboard import build_storyboard
from .util import PipelineError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="kidvid",
        description="Turn one prompt into an animated kids' story video or song.",
    )
    p.add_argument("prompt", help="the idea, e.g. 'a shy bunny who finds a friend'")
    p.add_argument("--title", default=None)
    p.add_argument("--scenes", type=int, default=4)
    p.add_argument("--seconds-per-scene", type=float, default=5.0)
    p.add_argument("--image", action="append", default=[],
                   help="starting image per scene; repeat for more scenes")
    p.add_argument("--no-music", action="store_true")
    p.add_argument("--narrate", action="store_true",
                   help="add offline narration / sung lyrics")
    p.add_argument("--video-backend", default=None,
                   choices=[None, "mock", "ai", "wangp"],
                   help="'ai' = free hosted image model, 'mock' = offline "
                        "gradients, 'wangp' = local GPU models")
    p.add_argument("--music-backend", default=None,
                   choices=[None, "auto", "mock", "ai", "acestep", "musicgen"],
                   help="'ai' = free hosted song model with vocals")
    p.add_argument("--dry-run", action="store_true",
                   help="print the storyboard and exit")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--seed", type=int, default=1234)

    args = p.parse_args(argv)

    if args.title:
        title = args.title
    else:
        # Title Case, trimmed at a word boundary so it never cuts mid-word.
        words = args.prompt.strip().rstrip(".!?").split()
        title_words: list[str] = []
        for word in words:
            if len(" ".join(title_words + [word])) > 34:
                break
            title_words.append(word)
        title = " ".join(title_words).title() or "My Story"

    cfg = ShowConfig(
        prompt=args.prompt,
        title=title,
        scene_count=args.scenes,
        seconds_per_scene=args.seconds_per_scene,
        image_paths=args.image,
        make_music=not args.no_music,
        narrate=args.narrate,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    cfg.music_seconds = cfg.scene_count * cfg.seconds_per_scene

    if args.dry_run:
        board = build_storyboard(cfg)
        print(f"Title:     {board.title}")
        print(f"Character: {board.character}")
        print(f"Music:     {board.music_prompt}")
        for scene in board.scenes:
            print(f"\n--- scene {scene.index + 1} [{scene.beat}] "
                  f"{scene.seconds:.1f}s ---")
            print(f"image:     {scene.image or '(text-to-video)'}")
            print(f"narration: {scene.narration}")
            print(f"prompt:    {scene.video_prompt}")
        if board.song:
            print("\n--- song ---")
            for line in board.song:
                print(f"  {line}")
        return 0

    try:
        make_video(cfg, video_backend=args.video_backend,
                   music_backend=args.music_backend, narrate=args.narrate)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
