"""Narration + song vocals via espeak-ng (fully offline, free, no GPU)."""

from __future__ import annotations

from pathlib import Path

from .util import run, have, PipelineError


class EspeakNarrator:
    """Offline TTS. Robotic but clear - ideal for a first working build.

    Swap in a neural voice later (Kokoro, Piper, XTTS) without touching
    the rest of the pipeline; only this class changes.
    """

    def __init__(self, voice: str = "en+f3", speed: int = 150) -> None:
        self.voice = voice
        self.speed = speed

    def available(self) -> bool:
        return have("espeak-ng") or have("espeak")

    def _binary(self) -> str:
        return "espeak-ng" if have("espeak-ng") else "espeak"

    def say(self, text: str, out_path: Path) -> Path:
        if not self.available():
            raise PipelineError(
                "espeak-ng not found. Install: sudo apt-get install -y espeak-ng"
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run([
            self._binary(), "-v", self.voice, "-s", str(self.speed),
            "-w", str(out_path), text,
        ])
        return out_path

    def sing_lines(self, lines: list[str], out_path: Path) -> Path:
        """Speak each line with a small gap so it reads as a chant."""
        if not self.available():
            raise PipelineError("espeak-ng not found")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        parts: list[Path] = []
        for i, line in enumerate(lines):
            part = out_path.with_name(f"{out_path.stem}_line{i}.wav")
            self.say(line, part)
            parts.append(part)

        from .movie import concat_audio

        concat_audio(parts, out_path, gap=0.25)
        for part in parts:
            part.unlink(missing_ok=True)
        return out_path
