"""Music generation: ACE-Step (fast, song-like) with MusicGen fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .util import run, probe_duration, PipelineError


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

    Install from GitHub, not PyPI. The PyPI ``ace-step`` sdist declares only
    ``packages=["acestep"]``, omitting the schedulers, models, music_dcae and
    language_segmentation subpackages its own pipeline imports, so it cannot
    import once installed. The repo also pins ``diffusers==0.32.2``, which
    predates Wan support and would break text-to-video; current main uses
    ``diffusers>=0.33.0`` and ``find_namespace_packages()``.

    Needs a GPU. On a T4 (no bfloat16) it runs in float32.
    """

    def __init__(self, steps: int = 60, checkpoint_dir: str | None = None) -> None:
        self.steps = steps
        self.checkpoint_dir = checkpoint_dir
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from acestep.pipeline_ace_step import ACEStepPipeline

            self._pipe = ACEStepPipeline(
                checkpoint_dir=self.checkpoint_dir,
                dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
                torch_compile=False,
            )
        return self._pipe

    def generate(self, prompt: str, seconds: float, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ACE-Step always writes WAV, ignoring the extension on save_path, and
        # the container is whatever `format` says. Ask for wav explicitly and
        # transcode afterwards, otherwise an .m4a path holds raw WAV bytes and
        # downstream muxing fails on a format it cannot parse.
        raw = out_path.with_suffix(".ace.wav")
        pipe = self._load()

        # Kaggle's T4 with the torch cu128 build: cuDNN's v8 graph API finds no
        # executable engine for the DC-AE decoder's conv2d and raises
        #     RuntimeError: GET was unable to find an engine to execute this
        #     computation
        # and it does so after diffusion has already finished, throwing away
        # the whole generation. Disabling cuDNN makes PyTorch use its native
        # conv kernels instead, which handle this shape. Scoped to this call
        # so the video path keeps cuDNN, where it is both faster and fine.
        import torch

        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            pipe(
                prompt=prompt,
                lyrics="",
                audio_duration=float(min(seconds, 240)),
                infer_step=self.steps,
                format="wav",
                save_path=str(raw),
            )
        finally:
            torch.backends.cudnn.enabled = prev_cudnn
        if not raw.exists():
            raise PipelineError(f"ACE-Step reported success but {raw} is missing")

        if out_path.suffix.lower() in (".wav", ""):
            raw.replace(out_path)
            return out_path

        run(["ffmpeg", "-y", "-i", str(raw), "-c:a", "aac", "-b:a", "192k",
             str(out_path)])
        raw.unlink(missing_ok=True)
        if not out_path.exists():
            raise PipelineError(f"failed to transcode {raw} to {out_path}")
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
    """Prefer ACE-Step (fast, Apache-2.0), fall back to MusicGen, then mock.

    ``import acestep`` succeeding is not sufficient: the PyPI ``ace-step``
    distribution installs a package that cannot import its own submodules, so
    constructing the backend is where that actually shows up. Callers that
    want a real song should treat a failure here as a real failure rather
    than silently accepting the mock melody.
    """
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
