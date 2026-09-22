"""Music generation: ACE-Step (fast, song-like) with MusicGen fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .util import run, probe_duration


class MusicBackend(Protocol):
    def generate(self, prompt: str, seconds: float, out_path: Path) -> Path: ...


class MockMusicBackend:
    """A gentle generated arpeggio - no model, no GPU.

    Lets you verify audio muxing and timing without downloading weights.
    Uses ffmpeg's sine sources tuned to a C-major pentatonic figure.
    """

    # C5 D5 E5 G5 A5
    _NOTES = (523.25, 587.33, 659.25, 783.99, 880.00)

    def generate(self, prompt: str, seconds: float, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # One sine voice per note, mixed down. Quiet by design so narration
        # or speech stays intelligible on top.
        pattern = [0, 2, 3, 4, 3, 2, 1, 0]
        note_len = 0.5
        cycles = max(1, int(seconds / (len(pattern) * note_len)) + 1)

        inputs: list[str] = []
        for i in range(cycles * len(pattern)):
            freq = self._NOTES[pattern[i % len(pattern)]]
            inputs += ["-f", "lavfi", "-i", f"sine=f={freq}:d={note_len}"]

        n = cycles * len(pattern)
        mixed = "".join(f"[{i}:a]" for i in range(n))
        filt = (
            f"{mixed}amix=inputs={n}:normalize=1,"
            f"volume=0.18,"
            f"afade=t=in:st=0:d=0.4,"
            f"afade=t=out:st={max(0.0, seconds - 0.6):.2f}:d=0.6,"
            f"atrim=0:{seconds:.2f}"
        )

        run([
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filt,
            "-c:a", "aac", "-b:a", "160k",
            str(out_path),
        ])
        return out_path


class ACEStepBackend:
    """ACE-Step: Apache-2.0 music model, fast (a few seconds on an A100).

    Requires GPU + the ACE-Step package installed in the notebook.
    """

    def __init__(self, steps: int = 60) -> None:
        self.steps = steps
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from acestep.pipeline_ace_step import ACEStepPipeline

            self._pipe = ACEStepPipeline(
                checkpoint_dir="checkpoints",
                dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
                torch_compile=False,
            )
        return self._pipe

    def generate(self, prompt: str, seconds: float, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pipe = self._load()
        pipe(
            prompt=prompt,
            lyrics="",
            audio_duration=float(min(seconds, 240)),
            infer_step=self.steps,
            save_path=str(out_path),
        )
        return out_path


class MusicGenBackend:
    """Meta MusicGen via audiocraft (MIT). Slower than ACE-Step, very reliable."""

    _SIZES = {"small": "facebook/musicgen-small",
              "medium": "facebook/musicgen-medium",
              "large": "facebook/musicgen-large"}

    def __init__(self, size: str = "small") -> None:
        self.size = size
        self._model = None

    def _load(self):
        if self._model is None:
            from audiocraft.models import MusicGen

            self._model = MusicGen.get_pretrained(self._SIZES[self.size])
            self._model.set_generation_params(duration=30)
        return self._model

    def generate(self, prompt: str, seconds: float, out_path: Path) -> Path:
        import torchaudio

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model = self._load()
        model.set_generation_params(duration=float(min(seconds, 30)))
        wav = model.generate([prompt])
        torchaudio.save(str(out_path), wav[0].cpu(), model.sample_rate)
        return out_path


def default_backend():
    """Prefer ACE-Step (fast, Apache-2.0), fall back to MusicGen, then mock."""
    try:
        import acestep  # noqa: F401

        return ACEStepBackend()
    except Exception:
        pass
    try:
        import audiocraft  # noqa: F401

        return MusicGenBackend()
    except Exception:
        return MockMusicBackend()
