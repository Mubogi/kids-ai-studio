"""Video generation backends.

`MockVideoBackend` needs no GPU and no model weights - it renders coloured
scene cards with ffmpeg. Use it to test the whole pipeline (storyboard ->
clips -> music -> final movie) before spending Kaggle GPU hours.

`WanGPBackend` drives the real open-source models. See wangp_backend.py for
the model list and VRAM notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import ShowConfig
from .storyboard import Scene
from .theme import DEFAULT_THEME, Theme
from .util import run, hex_hue


class VideoBackend(Protocol):
    def generate(self, scene: Scene, cfg: ShowConfig, out_path: Path) -> Path:
        """Produce one clip for `scene` at `out_path` (no audio)."""
        ...


class MockVideoBackend:
    """Renders a smooth colour-gradient clip per scene. No GPU required.

    Scene colour comes from the brand theme's per-beat palette, so each scene
    is visually distinct and you can confirm ordering in the final cut.
    """

    def __init__(self, seed: int = 0, theme: Theme | None = None) -> None:
        self.seed = seed
        self.theme = theme or DEFAULT_THEME

    def generate(self, scene: Scene, cfg: ShowConfig, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        base = self.theme.beats.get(scene.beat, self.theme.soft)
        drift = hex_hue(base, shift=(scene.index * 7) % 40)

        # Draw a soft vertical gradient in Python, then let ffmpeg loop it
        # into a real H.264 clip at the exact fps/duration/frame size.
        frame = out_path.with_suffix(".png")
        _draw_gradient(frame, cfg.width, cfg.height, base, drift)

        run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(frame),
            "-t", f"{scene.seconds:.3f}",
            "-r", str(cfg.fps),
            "-vf", f"scale={cfg.width}:{cfg.height},format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(out_path),
        ])
        frame.unlink(missing_ok=True)
        return out_path


def _draw_gradient(path: Path, w: int, h: int, top: str, bottom: str) -> None:
    """Write a cheap vertical gradient PNG. Falls back to a solid colour."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # No Pillow: build a 1x2 gradient with ffmpeg's own generator instead.
        run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"gradients=s={w}x{h}:c0={top}:c1={bottom}:d=1",
            "-frames:v", "1", str(path),
        ])
        return

    t = tuple(int(top.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    b = tuple(int(bottom.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        f = y / max(h - 1, 1)
        draw.line(
            [(0, y), (w, y)],
            fill=tuple(int(t[i] + (b[i] - t[i]) * f) for i in range(3)),
        )
    img.save(path, "PNG")
