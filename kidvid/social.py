"""Reframe one finished video for every social platform, and caption it.

Two jobs:

  * `to_platform`  - reframe to a platform's aspect ratio. Landscape source
    into a portrait slot is done with a blurred fill rather than a hard crop,
    because a centre crop of a 16:9 cartoon cuts the characters' heads off.
  * `auto_captions` - burn subtitles. The script is already known from the
    storyboard, so this needs no speech recognition and is frame-accurate
    against the narration that was actually recorded.

Everything here is ffmpeg, so it runs on a laptop with no GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ShowConfig
from .theme import DEFAULT_THEME, Theme, ffmpeg_color
from .util import run, probe_duration, PipelineError


@dataclass(frozen=True)
class Platform:
    key: str
    label: str
    width: int
    height: int
    max_seconds: float        # 0 = no practical limit
    note: str = ""


# The ratios and length habits each platform actually rewards.
PLATFORMS: dict[str, Platform] = {
    "tiktok": Platform("tiktok", "TikTok", 1080, 1920, 180,
                       "hook in the first 2 seconds"),
    "youtube_shorts": Platform("youtube_shorts", "YouTube Shorts", 1080, 1920, 60),
    "instagram_reels": Platform("instagram_reels", "Instagram Reels", 1080, 1920, 90),
    "youtube": Platform("youtube", "YouTube", 1920, 1080, 0),
    "instagram_feed": Platform("instagram_feed", "Instagram Feed", 1080, 1350, 60),
    "facebook": Platform("facebook", "Facebook", 1080, 1080, 240),
    "whatsapp_status": Platform("whatsapp_status", "WhatsApp Status", 1080, 1920, 30),
}


def _fill_filter(w: int, h: int, blur: int = 40) -> str:
    """Blurred background + letterboxed foreground - the standard reframe.

    Scales the source up to *cover* the frame, blurs it, then overlays the
    source scaled to *contain*. No content is ever cropped away.
    """
    return (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},boxblur={blur}:2,setsar=1[bg];"
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )


def to_platform(
    video: Path,
    platform: str | Platform,
    out_path: Path,
    *,
    style: str = "blur",
    theme: Theme | None = None,
) -> Path:
    """Reframe `video` for one platform. `style` is 'blur' or 'brand'."""
    p = PLATFORMS[platform] if isinstance(platform, str) else platform
    video, out_path = Path(video), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if style == "brand":
        theme = theme or DEFAULT_THEME
        colour = ffmpeg_color(theme.soft)
        vf = (
            f"scale={p.width}:{p.height}:force_original_aspect_ratio=decrease,"
            f"pad={p.width}:{p.height}:(ow-iw)/2:(oh-ih)/2:{colour},"
            f"format=yuv420p"
        )
        args = ["-vf", vf]
    else:
        args = ["-filter_complex", _fill_filter(p.width, p.height), "-map", "[v]"]

    cmd = ["ffmpeg", "-y", "-i", str(video), *args]

    if p.max_seconds:
        cmd += ["-t", f"{p.max_seconds:.3f}"]

    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        str(out_path),
    ]
    run(cmd)
    return out_path


def _ass_alpha(opacity: float) -> str:
    """ASS colours are &HAABBGGRR - alpha is inverted (00 = opaque)."""
    return f"{int(round((1.0 - max(0.0, min(opacity, 1.0))) * 255)):02X}"


def _srt_time(t: float) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def _wrap(text: str, width: int = 32) -> str:
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


def build_srt(storyboard_path: Path, out_path: Path, *, max_seconds: float = 0.0) -> Path:
    """Write subtitles from the storyboard's own narration and timings.

    No ASR: the spoken text is already known and the scene durations are
    exact, so captions land perfectly in sync.
    """
    board = json.loads(Path(storyboard_path).read_text())
    scenes = board.get("scenes", [])
    song = board.get("song") or []

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cues: list[tuple[float, float, str]] = []

    if song:
        # Spread lyric lines evenly across the runtime.
        total = max_seconds or sum(float(s.get("seconds", 0)) for s in scenes)
        per = total / max(len(song), 1)
        for i, line in enumerate(song):
            cues.append((i * per, (i + 1) * per - 0.05, line))
    else:
        clock = 0.0
        for s in scenes:
            dur = float(s.get("seconds", 0))
            text = (s.get("narration") or "").strip()
            if text:
                cues.append((clock + 0.1, clock + dur - 0.1, text))
            clock += dur

    blocks = []
    for i, (start, end, text) in enumerate(cues, 1):
        if end <= start:
            end = start + 0.8
        blocks.append(
            f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{_wrap(text)}\n"
        )
    out_path.write_text("\n".join(blocks))
    return out_path


# This ffmpeg's libass lays SRT subtitles out on a fixed 288-unit-tall
# virtual canvas regardless of the real frame size, so style values must be
# scaled by that ratio. The dedicated `ass` filter (which does honour
# PlayRes) fails to parse ASS files in this build, so SRT is the only
# reliable path - hence the scaling below rather than a PlayRes header.
_LIBASS_CANVAS_H = 288.0


def _scaled(value: float, height: int) -> int:
    return max(1, int(round(value * _LIBASS_CANVAS_H / height)))


def burn_captions(
    subtitle: Path,
    video: Path,
    out_path: Path,
    *,
    width: int = 0,
    height: int = 0,
    theme: Theme | None = None,
) -> Path:
    """Burn `subtitle` (SRT) into `video`, styled on-brand.

    Captions sit in the lower safe area, clear of the platform UI bars,
    on a dark translucent box so they stay readable over bright artwork.
    """
    subtitle, video, out_path = Path(subtitle), Path(video), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not width or not height:
        width, height = _frame_size(video)

    style = (
        f"FontName=DejaVu Sans,"
        f"Fontsize={_scaled(height * 0.042, height)},"
        f"PrimaryColour=&H00FFFFFF,"
        # BorderStyle=3 draws a box; Outline must stay >=1 or libass renders
        # nothing at all for these events.
        f"BorderStyle=3,Outline=1,Shadow=0,"
        f"BackColour=&H{_ass_alpha(0.72)}2B2117,"
        f"MarginV={_scaled(height * 0.060, height)},"
        f"MarginL={_scaled(width * 0.045, height)},"
        f"MarginR={_scaled(width * 0.045, height)},"
        f"Alignment=2"
    )
    esc = (str(subtitle).replace("\\", "/")
           .replace(":", r"\:").replace("'", r"\'"))
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"subtitles='{esc}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "copy", "-movflags", "+faststart",
        str(out_path),
    ])
    return out_path


def _frame_size(video: Path) -> tuple[int, int]:
    """Read width/height straight out of the container."""
    proc = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video),
    ])
    try:
        w, h = (int(x) for x in proc.stdout.strip().split(",")[:2])
        return w, h
    except Exception as e:  # noqa: BLE001
        raise PipelineError(
            f"could not read frame size of {video}: {proc.stdout!r}"
        ) from e


def export_all(
    video: Path,
    storyboard_path: Path,
    out_dir: Path,
    *,
    platforms: list[str] | None = None,
    style: str = "blur",
    captions: bool = True,
    theme: Theme | None = None,
) -> list[dict]:
    """One finished video in, a set of ready-to-upload platform cuts out.

    Returns a manifest so a UI can list exactly what was produced.
    """
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = platforms or list(PLATFORMS)
    source_seconds = probe_duration(video)

    srt: Path | None = None
    if captions:
        # The SRT doubles as a portable sidecar for uploads.
        srt = build_srt(storyboard_path, out_dir / "captions.srt",
                        max_seconds=source_seconds)

    manifest: list[dict] = []
    for key in keys:
        if key not in PLATFORMS:
            raise PipelineError(f"unknown platform: {key}")
        p = PLATFORMS[key]

        cut = to_platform(video, p, out_dir / f"{key}.mp4",
                          style=style, theme=theme)
        final = cut
        if srt:
            final = burn_captions(
                srt, cut, out_dir / f"{key}_captioned.mp4",
                width=p.width, height=p.height, theme=theme,
            )
            cut.unlink(missing_ok=True)

        manifest.append({
            "platform": key,
            "label": p.label,
            "file": str(final),
            "size": f"{p.width}x{p.height}",
            "duration": round(probe_duration(final), 2),
            "truncated": bool(p.max_seconds and source_seconds > p.max_seconds),
            "note": p.note,
            "captioned": srt is not None,
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
