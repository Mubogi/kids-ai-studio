"""End-to-end orchestration: prompt in, finished kids' video out."""

from __future__ import annotations

import json
import platform
from pathlib import Path

from .config import ShowConfig
from .storyboard import Storyboard, build_storyboard
from .util import require_ffmpeg, probe_duration, PipelineError
from . import movie


def _pick_video_backend(cfg: ShowConfig, backend: str | None):
    if backend == "mock" or backend is None and not _gpu_available():
        from .video import MockVideoBackend

        return MockVideoBackend(seed=cfg.seed)
    if backend in (None, "wangp"):
        from .wangp_backend import WanGPBackend

        return WanGPBackend()
    raise PipelineError(f"unknown video backend: {backend}")


def _pick_music_backend(backend: str | None):
    if backend == "mock":
        from .music import MockMusicBackend

        return MockMusicBackend()
    if backend in (None, "auto"):
        from .music import default_backend

        return default_backend()
    if backend == "acestep":
        from .music import ACEStepBackend

        return ACEStepBackend()
    if backend == "musicgen":
        from .music import MusicGenBackend

        return MusicGenBackend()
    raise PipelineError(f"unknown music backend: {backend}")


def _gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def make_video(
    cfg: ShowConfig,
    video_backend: str | None = None,
    music_backend: str | None = None,
    narrate: bool | None = None,
) -> dict:
    """Run the full pipeline. Returns a dict describing what was produced."""
    require_ffmpeg()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    board = build_storyboard(cfg)
    _write_storyboard(board, out_dir / "storyboard.json")

    vbackend = _pick_video_backend(cfg, video_backend)
    print(f"[1/5] storyboard: {len(board.scenes)} scenes, "
          f"{board.total_seconds():.1f}s total")
    print(f"      video backend: {type(vbackend).__name__}")

    # --- 1. clips -------------------------------------------------------
    clips: list[Path] = []
    for scene in board.scenes:
        target = clips_dir / f"scene_{scene.index:02d}.mp4"
        kind = "image-to-video" if scene.image else "text-to-video"
        print(f"      scene {scene.index + 1}/{len(board.scenes)} ({kind}) ...")
        vbackend.generate(scene, cfg, target)
        clips.append(target)

    # --- 2. join --------------------------------------------------------
    print("[2/5] joining clips")
    silent = movie.concat_clips(clips, cfg, out_dir / "video_silent.mp4")

    # --- 3. music -------------------------------------------------------
    audio: Path | None = None
    if cfg.make_music:
        print("[3/5] generating music")
        total = probe_duration(silent)
        mbackend = _pick_music_backend(music_backend)
        print(f"      music backend: {type(mbackend).__name__}")
        audio = mbackend.generate(
            board.music_prompt, min(cfg.music_seconds, total), out_dir / "music.m4a"
        )
    else:
        print("[3/5] music disabled")

    # --- 4. narration ---------------------------------------------------
    voiceover: Path | None = None
    want_narration = cfg.narrate if narrate is None else narrate
    if want_narration:
        print("[4/5] narrating")
        from .tts import EspeakNarrator

        narrator = EspeakNarrator(voice=cfg.voice)
        if not narrator.available():
            print("      espeak-ng missing, skipping narration")
        elif board.song:
            voiceover = narrator.sing_lines(board.song, out_dir / "voice.wav")
        else:
            parts: list[Path] = []
            for scene in board.scenes:
                part = out_dir / f"line_{scene.index:02d}.wav"
                narrator.say(scene.narration, part)
                parts.append(part)
            voiceover = movie.concat_audio(parts, out_dir / "voice.wav", gap=0.3)
            for part in parts:
                part.unlink(missing_ok=True)
    else:
        print("[4/5] narration disabled")

    final_audio = _mix_audio(audio, voiceover, out_dir) if (audio or voiceover) else None

    # --- 5. mux + polish ------------------------------------------------
    print("[5/5] finishing")
    finished = movie.mux(silent, final_audio, out_dir / "final.mp4", cfg)

    if board.song:
        captioned = movie.add_song_captions(
            finished, board.song, offset=0.0,
            out_path=out_dir / "final_captioned.mp4", cfg=cfg,
        )
        finished = captioned
    elif cfg.title:
        finished = movie.burn_title(
            finished, cfg.title, out_dir / "final_titled.mp4", cfg
        )

    result = {
        "video": str(finished),
        "storyboard": str(out_dir / "storyboard.json"),
        "scenes": len(board.scenes),
        "duration": round(probe_duration(finished), 2),
        "video_backend": type(vbackend).__name__,
        "has_music": audio is not None,
        "has_narration": voiceover is not None,
        "is_song": bool(board.song),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def _mix_audio(music: Path | None, voice: Path | None, out_dir: Path) -> Path | None:
    """Duck the music under narration so words stay clear."""
    from .util import run

    if music and not voice:
        return music
    if voice and not music:
        return voice

    out = out_dir / "mixed.m4a"
    run([
        "ffmpeg", "-y", "-i", str(voice), "-i", str(music),
        "-filter_complex",
        "[1:a]volume=0.35[bed];[0:a][bed]amix=inputs=2:duration=longest:normalize=0[a]",
        "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(out),
    ])
    return out


def _write_storyboard(board: Storyboard, path: Path) -> None:
    payload = {
        "title": board.title,
        "character": board.character,
        "theme": board.theme,
        "music_prompt": board.music_prompt,
        "song": board.song,
        "scenes": [
            {
                "index": s.index,
                "beat": s.beat,
                "seconds": s.seconds,
                "narration": s.narration,
                "video_prompt": s.video_prompt,
                "image": s.image,
            }
            for s in board.scenes
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
