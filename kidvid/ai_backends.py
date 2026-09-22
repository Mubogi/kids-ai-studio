"""Free AI backends - no GPU and no model download.

Three hosted services do the actual work:

  * Stable Horde  - text-to-image stills, keyless (anonymous key)
  * ACE-Step      - text-to-song *with vocals*, via a Hugging Face Space
  * edge-tts      - neural narration via Edge's read-aloud endpoint

The rule every backend here follows: degrade, never destroy. A sleeping
Space or a dropped connection raises PipelineError so the caller can fall
back, rather than leaving the user with a half-written video and no clue.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote

from .config import ShowConfig
from .storyboard import Scene
from .util import run, PipelineError


def _run_async(coro):
    """Run a coroutine to completion from sync code.

    `asyncio.run` cannot be used inside a Jupyter notebook: the kernel already
    has a running event loop, and asyncio refuses to nest one. That is not a
    corner case here, since the Kaggle notebook is the primary way this
    package is run, and it broke narration there while working on the CLI.

    So when a loop is already running, run the coroutine on a fresh loop in a
    worker thread instead. Threads are allowed to have their own loops.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running: the simple path is correct.
        return asyncio.run(coro)

    result: list = []
    error: list = []

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result.append(loop.run_until_complete(coro))
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            error.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if error:
        raise error[0]
    return result[0] if result else None


# Pollinations moved to a keyed, pre-paid model in 2026: anonymous requests
# now return 401/402 ("Insufficient balance"), so it is kept only as a
# fallback for anyone who supplies their own key. Stable Horde replaced it
# as the keyless option - see HordeImageBackend.
POLLINATIONS = "https://image.pollinations.ai/prompt/"
HORDE_API = "https://stablehorde.net/api/v2"
# The documented anonymous key. Real but shared, so queue priority is low.
HORDE_ANON_KEY = "0000000000"
ACESTEP_SPACE = "ace-step/ACE-Step"

# Edge's most expressive child voice - right register for picture-book read-alongs.
KID_VOICE = "en-US-AnaNeural"


def _http_get(url: str, dest: Path, *, attempts: int = 3, timeout: int = 90) -> Path:
    """Fetch `url` to `dest`, retrying transient failures.

    Image services queue aggressively under load and will time out or 500 on
    a cold request, so a single attempt is not enough to call one unreliable.
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


def _still_to_motion(still: Path, scene: Scene, cfg: ShowConfig, out_path: Path) -> Path:
    """Turn one generated still into a clip with a slow camera move.

    Shared by every image backend so the motion looks identical whichever
    service drew the picture. The move alternates push-in / pull-out so
    consecutive scenes do not read as one long drift.
    """
    frames = max(int(scene.seconds * cfg.fps), 1)
    if scene.index % 2 == 0:
        zoom = "min(zoom+0.0012,1.18)"
    else:
        zoom = "if(lte(zoom,1.0),1.18,max(1.001,zoom-0.0012))"

    # zoompan is smoother when it works on a frame at least as large as the
    # output, so scale up first and let the crop take the final frame.
    from .util import probe_size

    src_w, src_h = probe_size(still)
    scale = max(cfg.width / src_w, cfg.height / src_h, 1.0)
    scaled_w, scaled_h = int(src_w * scale) + 2, int(src_h * scale) + 2

    run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(still),
        "-vf",
        f"scale={scaled_w}:{scaled_h}:force_original_aspect_ratio=increase,"
        f"crop={cfg.width}:{cfg.height},"
        f"zoompan=z='{zoom}':d={frames}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={cfg.width}x{cfg.height}:fps={cfg.fps},format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-t", f"{scene.seconds:.3f}",
        str(out_path),
    ])
    return out_path


class HordeImageBackend:
    """Keyless text-to-image via Stable Horde's volunteer GPU network.

    This replaces Pollinations, which now requires a paid key. Stable Horde
    still accepts the documented anonymous key, so no signup is needed. The
    trade-off is a shared queue: anonymous jobs wait for spare volunteer
    capacity, typically under a minute but occasionally several.

    Pass a personal key from stablehorde.net to jump the queue - it is free
    and kudos-based, earned by donating GPU time.
    """

    def __init__(self, api_key: str = HORDE_ANON_KEY, timeout: int = 90,
                 max_wait: int = 420, models: list[str] | None = None) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_wait = max_wait
        # Omit `models` so any worker can serve it; naming "stable_diffusion"
        # restricts us to one older model and lengthens the queue.
        self.models = models

    def available(self) -> bool:
        import requests

        try:
            r = requests.get(f"{HORDE_API}/status/models", timeout=15,
                             params={"type": "image"})
            return r.status_code == 200
        except Exception:  # noqa: BLE001 - reports unavailable, not fatal
            return False

    def generate(self, scene: Scene, cfg: ShowConfig, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Horde tops out around 1024 wide on the common models. Request the
        # output size directly so no scaling is needed afterwards.
        width, height = self._clamp(cfg.width, cfg.height)

        payload: dict = {
            "prompt": self._prompt_for(scene, cfg),
            "params": {
                "width": width,
                "height": height,
                "steps": cfg.steps,
                "cfg_scale": cfg.guidance,
                "n": 1,
                "sampler_name": "k_euler",
            },
            "nsfw": False,
            "censor_nsfw": True,
            "trusted_workers": False,
            "slow_workers": True,
        }
        if self.models:
            payload["models"] = self.models

        import requests

        headers = {"apikey": self.api_key, "Content-Type": "application/json"}
        seed = cfg.seed + scene.index * 97
        payload["params"]["seed"] = str(seed)

        try:
            r = requests.post(f"{HORDE_API}/generate/async", json=payload,
                              headers=headers, timeout=self.timeout)
            r.raise_for_status()
            job_id = r.json()["id"]
        except Exception as e:  # noqa: BLE001 - surfaced as PipelineError
            raise PipelineError(f"stable horde rejected the job: {e}") from e

        img_url = self._wait_for_image(job_id)
        still = out_path.with_suffix(".webp")
        _http_get(img_url, still, attempts=3, timeout=self.timeout)
        try:
            _still_to_motion(still, scene, cfg, out_path)
        finally:
            still.unlink(missing_ok=True)
        return out_path

    def _wait_for_image(self, job_id: str) -> str:
        """Poll until the job finishes, then return the signed image URL."""
        import requests

        deadline = time.time() + self.max_wait
        headers = {"apikey": self.api_key}
        last_state = ""
        while time.time() < deadline:
            try:
                r = requests.get(f"{HORDE_API}/generate/check/{job_id}",
                                 headers=headers, timeout=30)
                r.raise_for_status()
                info = r.json()
            except Exception:  # noqa: BLE001 - transient, keep polling
                time.sleep(5)
                continue

            if info.get("faulted"):
                raise PipelineError(f"stable horde faulted the job: {info}")
            if info.get("done"):
                break

            # `is_possible: false` means no worker can ever serve this
            # request - waiting the full timeout would be pointless.
            if info.get("is_possible") is False:
                raise PipelineError(
                    "stable horde has no worker for this request: "
                    f"{info.get('message', info)}"
                )

            wait = info.get("wait_time")
            state = f"queue={info.get('queue_position')} eta={wait}s"
            if state != last_state:
                print(f"      horde: {state}")
                last_state = state
            time.sleep(5)
        else:
            raise PipelineError(
                f"stable horde did not finish within {self.max_wait}s"
            )

        try:
            s = requests.get(f"{HORDE_API}/generate/status/{job_id}",
                             headers=headers, timeout=60)
            s.raise_for_status()
            gens = s.json().get("generations") or []
        except Exception as e:  # noqa: BLE001
            raise PipelineError(f"could not read horde result: {e}") from e

        if not gens:
            raise PipelineError("stable horde returned no images")
        # Take the full URL including its signature - truncating it breaks
        # the download with a 400 Authorization error.
        return gens[0]["img"]

    @staticmethod
    def _prompt_for(scene: Scene, cfg: ShowConfig) -> str:
        """Single-line prompt. Horde has no separate negative field here, so
        the style suffix carries the whole look."""
        return " ".join(scene.video_prompt.split())

    @staticmethod
    def _clamp(width: int, height: int) -> tuple[int, int]:
        """Horde rejects odd sizes and very large ones; keep to multiples of
        64 inside a sane cap and preserve the aspect ratio."""
        cap_w, cap_h = 1024, 1024
        scale = min(cap_w / width, cap_h / height, 1.0)
        w = max(int(width * scale) // 64 * 64, 64)
        h = max(int(height * scale) // 64 * 64, 64)
        return w, h


class ImageVideoBackend:
    """Pollinations text-to-image per scene, moved with a slow camera push.

    Kept for anyone with a Pollinations key, but the anonymous free tier is
    gone, so prefer HordeImageBackend.
    """

    def __init__(self, model: str = "flux", timeout: int = 90,
                 api_key: str | None = None) -> None:
        self.model = model
        self.timeout = timeout
        # Pollinations now bills per request; pass a token to use it.
        self.api_key = api_key

    def generate(self, scene: Scene, cfg: ShowConfig, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        prompt = " ".join(scene.video_prompt.split())
        if len(prompt) > 420:
            prompt = prompt[:420].rsplit(" ", 1)[0]

        # Never ask for more than the service's 1024x576 cap: it downscales,
        # and the old code then upscaled that back, which is what made
        # earlier frames look soft.
        gen_w, gen_h = self._native_size(cfg.width, cfg.height)

        seed = cfg.seed + scene.index * 97
        url = (
            f"{POLLINATIONS}{quote(prompt, safe='')}"
            f"?width={gen_w}&height={gen_h}"
            f"&model={self.model}&nologo=true&seed={seed}"
        )
        if self.api_key:
            url += f"&token={self.api_key}"

        still = out_path.with_suffix(".jpg")
        _http_get(url, still, timeout=self.timeout)
        try:
            _still_to_motion(still, scene, cfg, out_path)
        finally:
            still.unlink(missing_ok=True)
        return out_path

    @staticmethod
    def _native_size(width: int, height: int) -> tuple[int, int]:
        """Largest size the image service actually honours for this aspect.

        It tops out at 1024x576 and preserves the requested aspect ratio by
        fitting inside it, so asking for more is wasted.
        """
        max_w, max_h = 1024, 576
        scale = min(max_w / width, max_h / height, 1.0)
        return max(int(width * scale), 64), max(int(height * scale), 64)


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
            _run_async(_speak())
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
