"""Small shell / colour helpers shared across the package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising PipelineError with captured output on failure."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-25:])
        raise PipelineError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:6])}...\n{tail}"
        )
    if not quiet and proc.stdout:
        print(proc.stdout, file=sys.stderr)
    return proc


def have(binary: str) -> bool:
    from shutil import which

    return which(binary) is not None


def require_ffmpeg() -> None:
    if not have("ffmpeg") or not have("ffprobe"):
        raise PipelineError(
            "ffmpeg and ffprobe are required. "
            "Install with: sudo apt-get install -y ffmpeg"
        )


def probe_duration(path: Path) -> float:
    """Duration of a media file in seconds."""
    proc = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise PipelineError(f"could not read duration of {path}") from exc


def hex_hue(hex_color: str, shift: int = 0) -> str:
    """Lighten a hex colour by a 0-100 amount. Used to vary scene colours."""
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    lifted = [min(255, c + int((255 - c) * shift / 100)) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in lifted)
