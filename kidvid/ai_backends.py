"""Free AI backends - no GPU, no API key, no model download.

Three hosted services do the actual work:

  * Pollinations  - text-to-image stills, keyless
  * ACE-Step      - text-to-song *with vocals*, via a Hugging Face Space
  * edge-tts      - neural narration via Edge's read-aloud endpoint

The rule every backend here follows: degrade, never destroy. A sleeping
Space or a dropped connection raises PipelineError so the caller can fall
back, rather than leaving the user with a half-written video and no clue.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from urllib.parse import quote

from .config import ShowConfig
from .storyboard import SAFE_STYLE, Scene
from .util import run, PipelineError

POLLINATIONS = "https://image.pollinations.ai/prompt/"
ACESTEP_SPACE = "ace-step/ACE-Step"

# Edge's most expressive child voice - right register for picture-book read-alongs.
KID_VOICE = "en-US-AnaNeural"


def _http_get(url: str, dest: Path, *, attempts: int = 3, timeout: int = 90) -> Path:
    """Fetch `url` to `dest`, retrying transient failures.

    Pollinations queues aggressively under load and will time out on a cold
    request, so a single attempt is not enough to call it unreliable.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for i in range(attempts):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                body = r.content
            if len(body) < 1024:
                raise PipelineError(f"response too small ({len(body)} bytes)")
            dest.write_bytes(body)
            return dest
        except Exception as e:  # noqa: BLE001 - retried below
            last = e
            if i + 1 < attempts:
                time.sleep(2 * (i + 1))
    raise PipelineError(f"could not fetch image after {attempts} tries: {last}")


class ImageVideoBackend:
    """Real text-to-image per scene, moved with a slow camera push.

    Pollinations draws the still; ffmpeg adds the Ken Burns move. Be clear
    about what this is: genuine AI artwork per scene, but the motion is
    camera-only. Diffusion *video* needs a real GPU (see wangp_backend).
    """

    def __init__(self, model: str = "flux", timeout: int = 90) -> None:
        self.model = model
        self.timeout = timeout

    def generate(self, scene: Scene, cfg: ShowConfig, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # scene.video_prompt already carries SAFE_STYLE; keep it under URL
        # limits without dropping the kid-safe tail.
        prompt = scene.video_prompt.strip()
        if len(prompt) > 420:
            prompt = prompt[:420].rsplit(" ", 1)[0]

        seed = cfg.seed + scene.index * 97
        url = (
            f"{POLLINATIONS}{quote(prompt, safe='')}"
            f"?width={cfg.width}&height={cfg.height}"
            f"&model={self.model}&nologo=true&seed={seed}"
        )

        still = out_path.with_suffix(".jpg")
        _http_get(url, still, timeout=self.timeout)

        frames = max(int(scene.seconds * cfg.fps), 1)
        # Alternate push-in / pull-out so consecutive scenes do not read as
        # one long drift.
        if scene.index % 2 == 0:
            zoom = "min(zoom+0.0012,1.18)"
        else:
            zoom = "if(lte(zoom,1.0),1.18,max(1.001,zoom-0.0012))"

        run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(still),
            "-vf",
            f"scale={cfg.width * 2}:{cfg.height * 2},"
            f"zoompan=z='{zoom}':d={frames}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={cfg.width}x{cfg.height}:fps={cfg.fps},format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-t", f"{scene.seconds:.3f}",
            str(out_path),
        ])
        still.unlink(missing_ok=True)
        return out_path


class ACEStepMusicBackend:
    """Real AI song generation, vocals included, on a free HF Space.

    Unlike the mock arpeggio this sings the storyboard's lyrics, which is
    what makes song mode sound like a song instead of a tone.
    """

    def __init__(self, space: str = ACESTEP_SPACE, steps: int = 30) -> None:
        self.space = space
        self.steps = steps

    def available(self) -> bool:
        try:
            import gradio_client  # noqa: F401

            return True
        except ImportError:
            return False

    def generate(self, prompt: str, seconds: float, out_path: Path,
                 lyrics: str | None = None) -> Path:
        if not self.available():
            raise PipelineError(
                "gradio_client not installed. pip install gradio_client"
            )
        from gradio_client import Client

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # The Space returns one of a few containers depending on version.
        if lyrics and lyrics.strip():
            text = lyrics.strip()
        else:
            text = "[instrumental]"

        try:
            client = Client(self.space, verbose=False)
            result = client.predict(
                audio_duration=int(round(seconds)),
                prompt=prompt,
                lyrics=text,
                infer_step=self.steps,
                guidance_scale=15,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10,
                manual_seeds="[42]",
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3,
                api_name="/__call__",
            )
        except Exception as e:  # noqa: BLE001 - Space may be cold or down
            raise PipelineError(f"ACE-Step unavailable: {e}") from e

        produced = _first_audio_path(result)
        if produced is None:
            raise PipelineError(f"ACE-Step returned no audio: {result!r}")

        shutil.copyfile(produced, out_path)
        return out_path


def _first_audio_path(result) -> Path | None:
    """Dig the generated file out of whatever shape gradio returned."""
    seen: list = [result]
    while seen:
        item = seen.pop(0)
        if isinstance(item, str):
            if item.lower().endswith((".mp3", ".wav", ".flac", ".ogg")):
                p = Path(item)
                if p.exists():
                    return p
        elif isinstance(item, dict):
            seen.extend(item.values())
        elif isinstance(item, (list, tuple)):
            seen.extend(item)
    return None


class EdgeNarrator:
    """Neural narration, same call shape as EspeakNarrator.

    Swap-in compatible with the pipeline's narrator interface so nothing
    downstream needs to know which engine spoke.
    """

    def __init__(self, voice: str = KID_VOICE, rate: str = "+0%") -> None:
        self.voice = voice
        self.rate = rate

    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401

            return True
        except ImportError:
            return False

    def say(self, text: str, out_path: Path) -> Path:
        if not self.available():
            raise PipelineError("edge-tts not installed. pip install edge-tts")
        import edge_tts

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".mp3")

        async def _speak() -> None:
            await edge_tts.Communicate(text, self.voice, rate=self.rate).save(str(tmp))

        try:
            asyncio.run(_speak())
        except Exception as e:  # noqa: BLE001 - network / voice hiccup
            tmp.unlink(missing_ok=True)
            raise PipelineError(f"edge-tts failed: {e}") from e

        # Downstream concat assumes uncompressed wav.
        run([
            "ffmpeg", "-y", "-i", str(tmp),
            "-ar", "44100", "-ac", "1", str(out_path),
        ])
        tmp.unlink(missing_ok=True)
        return out_path

    def sing_lines(self, lines: list[str], out_path: Path) -> Path:
        """Speak each line with a beat between so it reads as a chant."""
        out_path = Path(out_path)
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
