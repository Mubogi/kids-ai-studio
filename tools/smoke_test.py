"""Verify the notebook's non-GPU logic runs exactly as written.

Run: python3 tools/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kidvid.config import ShowConfig
from kidvid.storyboard import build_storyboard
from kidvid.movie import concat_clips, mux, add_song_captions, burn_title
from kidvid.util import probe_duration
from kidvid.video import MockVideoBackend
from kidvid.music import MockMusicBackend

CASES = [
    ("story", "a brave little turtle who learns to share"),
    ("song", "a song about a friendly dragon"),
    ("picture", "my dog goes to space"),
]

out = Path("/tmp/kidvid_smoke")
out.mkdir(exist_ok=True)

for label, prompt in CASES:
    cfg = ShowConfig(
        prompt=prompt, title=prompt.title()[:34], scene_count=4,
        seconds_per_scene=2.0, out_dir=str(out / label), narrate=False,
    )
    cfg.music_seconds = 8
    board = build_storyboard(cfg)
    assert board.character, f"{label}: no character found"

    vb = MockVideoBackend()
    clips = [
        vb.generate(s, cfg, Path(cfg.out_dir) / "clips" / f"s{s.index}.mp4")
        for s in board.scenes
    ]
    silent = concat_clips(clips, cfg, Path(cfg.out_dir) / "silent.mp4")
    music = MockMusicBackend().generate(board.music_prompt, 8, Path(cfg.out_dir) / "m.m4a")
    final = mux(silent, music, Path(cfg.out_dir) / "final.mp4", cfg)
    final = (
        add_song_captions(final, board.song, 0.0, Path(cfg.out_dir) / "cap.mp4", cfg)
        if board.song
        else burn_title(final, cfg.title, Path(cfg.out_dir) / "t.mp4", cfg)
    )

    duration = probe_duration(final)
    assert 7.5 < duration < 8.5, f"{label}: unexpected duration {duration}"
    print(f"  {label:8} char={board.character!r:24} "
          f"scenes={len(board.scenes)} dur={duration:.1f}s OK")

print("\nALL SMOKE TESTS PASSED")
