"""Assemble clips, music, and narration into one finished movie."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .util import run, probe_duration, PipelineError


def _uniformize(clip: Path, cfg, dst: Path) -> Path:
    """Re-encode one clip to a single known fps/size/pixel format.

    The concat demuxer requires identical streams; models happily emit
    slightly different frame rates per call, so normalise first.
    """
    run([
        "ffmpeg", "-y", "-i", str(clip),
        "-vf", f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=decrease,"
               f"pad={cfg.width}:{cfg.height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={cfg.fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ])
    return dst


def concat_clips(clips: list[Path], cfg, out_path: Path) -> Path:
    """Join clips in order into one silent video track."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        normalised = [
            _uniformize(Path(c), cfg, tmp_dir / f"n{i:03d}.mp4")
            for i, c in enumerate(clips)
        ]
        listfile = tmp_dir / "clips.txt"
        listfile.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in normalised)
        )
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile), "-c", "copy", str(out_path),
        ])
    return out_path


def concat_audio(parts: list[Path], out_path: Path, gap: float = 0.0) -> Path:
    """Join audio files in order, optionally pausing between each."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise PipelineError("concat_audio needs at least one input")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        staged: list[Path] = []
        for i, part in enumerate(parts):
            wav = tmp_dir / f"a{i:03d}.wav"
            run(["ffmpeg", "-y", "-i", str(part), "-ar", "24000", "-ac", "1", str(wav)])
            staged.append(wav)
            if gap > 0 and i < len(parts) - 1:
                silence = tmp_dir / f"g{i:03d}.wav"
                run([
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"anullsrc=r=24000:cl=mono",
                    "-t", f"{gap:.3f}", str(silence),
                ])
                staged.append(silence)

        listfile = tmp_dir / "audio.txt"
        listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in staged))
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile), "-c:a", "pcm_s16le", str(out_path),
        ])
    return out_path


def mux(video: Path, audio: Path | None, out_path: Path, cfg) -> Path:
    """Attach an audio track to a video, trimming to the shorter of the two."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if audio is None:
        shutil.copyfile(video, out_path)
        return out_path

    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ])
    return out_path


def burn_title(video: Path, title: str, out_path: Path, cfg) -> Path:
    """Overlay a JD Hub branded title card for the first 3 seconds."""
    from .theme import DEFAULT_THEME, ffmpeg_color, attribution

    theme = DEFAULT_THEME
    out_path = Path(out_path)
    safe = title.replace(":", "\\:").replace("'", "\u2019")
    brand = attribution().replace(":", "\\:").replace("'", "\u2019")

    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fontarg = f":fontfile={font}" if Path(font).exists() else ""

    title_size = max(20, cfg.height // 12)
    brand_size = max(11, cfg.height // 30)

    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf",
        # Title band
        f"drawtext=text='{safe}':fontcolor={ffmpeg_color(theme.ink)}:fontsize={title_size}"
        f"{fontarg}:box=1:boxcolor={ffmpeg_color(theme.primary, 0.92)}:boxborderw=22:"
        f"x=(w-text_w)/2:y=h*0.72:enable='between(t,0,3)',"
        # Brand line under it
        f"drawtext=text='{brand}':fontcolor=white:fontsize={brand_size}"
        f"{fontarg}:box=1:boxcolor={ffmpeg_color(theme.deep, 0.85)}:boxborderw=12:"
        f"x=(w-text_w)/2:y=h*0.72+{title_size + 34}:enable='between(t,0,3)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(out_path),
    ])
    return out_path


def add_song_captions(video: Path, lines: list[str], offset: float, out_path: Path, cfg) -> Path:
    """Burn lyric lines as themed captions - big and readable for kids."""
    from .theme import DEFAULT_THEME, ffmpeg_color, attribution

    theme = DEFAULT_THEME
    out_path = Path(out_path)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fontarg = f":fontfile={font}" if Path(font).exists() else ""

    total = probe_duration(video)
    per_line = max(0.8, (total - offset) / max(len(lines), 1))
    size = max(16, cfg.height // 18)

    filters = []
    for i, line in enumerate(lines):
        start = offset + i * per_line
        end = min(total, start + per_line)
        safe = line.replace(":", "\\:").replace("'", "\u2019").replace("%", "\\%")
        filters.append(
            f"drawtext=text='{safe}':fontcolor={ffmpeg_color(theme.ink)}:fontsize={size}"
            f"{fontarg}:box=1:boxcolor={ffmpeg_color(theme.primary, 0.88)}:boxborderw=16:"
            f"x=(w-text_w)/2:y=h*0.85:enable='between(t,{start:.2f},{end:.2f})'"
        )

    if not filters:
        shutil.copyfile(video, out_path)
        return out_path

    # Brand tag in the corner for the whole song.
    filters.append(
        f"drawtext=text='{attribution()}':fontcolor={ffmpeg_color(theme.deep)}:"
        f"fontsize={max(10, cfg.height // 40)}{fontarg}:"
        f"x=w-text_w-16:y=16:enable='gte(t,0)'"
    )

    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(out_path),
    ])
    return out_path
