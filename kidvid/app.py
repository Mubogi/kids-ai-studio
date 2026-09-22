"""JD Hub Kids AI Studio - the web interface.

Type a prompt, press Generate, get a video back. The heavy lifting stays in
`kidvid.pipeline`; this module is the queue, the progress reporting and the
static file serving around it.

Jobs run one at a time on a background worker. The free hosted models behind
`--video-backend ai` do not enjoy being hammered in parallel, and a single
worker keeps progress reporting honest instead of interleaving several jobs'
log lines into one confusing stream.
"""

from __future__ import annotations

import contextlib
import io
import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .config import ShowConfig
from .pipeline import make_video
from .social import PLATFORMS, export_all
from .theme import BRAND, BRAND_LONG, CONTACT, WHATSAPP
from .util import PipelineError

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
WORK_DIR = ROOT / "output" / "studio"

app = FastAPI(title=BRAND, docs_url=None, redoc_url=None)


@dataclass
class Job:
    id: str
    request: dict
    status: str = "queued"          # queued | running | done | error
    stage: str = "Waiting in queue"
    percent: int = 0
    logs: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    social: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "percent": self.percent,
            "logs": self.logs[-40:],
            "result": self.result,
            "error": self.error,
            "social": self.social,
            "request": self.request,
        }


JOBS: dict[str, Job] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()
_LOCK = threading.Lock()

# "[3/5] generating music" -> 3 of 5 -> 40%..100% band
_STAGE_RE = re.compile(r"^\[(\d)/5\]")


class _ProgressWriter(io.TextIOBase):
    """Captures the pipeline's stdout and turns it into progress events."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.buf = ""

    def write(self, text: str) -> int:
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.rstrip()
            if not line:
                continue
            with _LOCK:
                self.job.logs.append(line)
                m = _STAGE_RE.match(line)
                if m:
                    step = int(m.group(1))
                    self.job.percent = int((step - 1) / 5 * 100)
                    self.job.stage = line.split("]", 1)[-1].strip().capitalize()
        return len(text)

    def flush(self) -> None:  # required by TextIOBase
        pass


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    req = job.request
    out_dir = WORK_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with _LOCK:
        job.status = "running"
        job.stage = "Writing the storyboard"
        job.percent = 2

    cfg = ShowConfig(
        prompt=req["prompt"],
        title=req["title"] or "My Story",
        scene_count=req["scenes"],
        seconds_per_scene=req["seconds_per_scene"],
        image_paths=req.get("images", []),
        art_style=req.get("art_style", "cartoon"),
        make_music=req.get("music", True),
        narrate=req.get("narrate", True),
        out_dir=str(out_dir),
        seed=req.get("seed", 1234),
    )
    cfg.music_seconds = cfg.scene_count * cfg.seconds_per_scene

    writer = _ProgressWriter(job)
    try:
        with contextlib.redirect_stdout(writer):
            result = make_video(
                cfg,
                video_backend=req["video_backend"],
                music_backend=req["music_backend"],
                narrate=cfg.narrate,
            )
        if writer.buf.strip():
            with _LOCK:
                job.logs.append(writer.buf.strip())

        with _LOCK:
            job.percent = 100
            job.stage = "Ready"
            job.result = {
                "video": f"/media/{job_id}/{Path(result['video']).name}",
                "storyboard": f"/media/{job_id}/storyboard.json",
                "scenes": result["scenes"],
                "duration": result["duration"],
                "video_backend": result["video_backend"],
                "has_music": result["has_music"],
                "has_narration": result["has_narration"],
                "is_song": result["is_song"],
                "title": cfg.title,
            }
            job.status = "done"
    except PipelineError as exc:
        with _LOCK:
            job.status = "error"
            job.error = str(exc)
            job.stage = "Failed"
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected
        with _LOCK:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.stage = "Failed"
            job.logs.append(traceback.format_exc()[-1500:])


def _worker() -> None:
    while True:
        job_id = QUEUE.get()
        try:
            _run_job(job_id)
        finally:
            QUEUE.task_done()


threading.Thread(target=_worker, daemon=True, name="kidvid-worker").start()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text())


@app.get("/api/platforms")
def platforms() -> dict:
    return {
        "brand": BRAND,
        "brand_long": BRAND_LONG,
        "contact": CONTACT,
        "whatsapp": WHATSAPP,
        "platforms": [
            {"key": p.key, "label": p.label, "size": f"{p.width}x{p.height}",
             "max_seconds": p.max_seconds, "note": p.note}
            for p in PLATFORMS.values()
        ],
    }


@app.post("/api/jobs")
async def create_job(
    prompt: str = Form(...),
    title: str = Form(""),
    mode: str = Form("story"),              # story | song
    video_backend: str = Form("ai"),        # ai | mock
    music_backend: str = Form("ai"),        # ai | mock
    art_style: str = Form("cartoon"),       # cartoon | cinematic | 3d | anime
    scenes: int = Form(3),
    seconds_per_scene: float = Form(4.0),
    music: bool = Form(True),
    narrate: bool = Form(True),
    seed: int = Form(1234),
    images: list[UploadFile] = File(default=[]),
) -> dict:
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Please describe the video you want.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for i, upload in enumerate(images or []):
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix.lower() or ".png"
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        dest = job_dir / f"input_{i:02d}{suffix}"
        dest.write_bytes(await upload.read())
        saved.append(str(dest))

    # Song mode asks the storyboard for lyrics and lets the music model sing
    # them; story mode narrates prose.
    is_song = mode == "song"
    final_title = title.strip() or ("A Song" if is_song else "My Story")

    job = Job(
        id=job_id,
        request={
            "prompt": prompt,
            "title": final_title,
            "mode": mode,
            "video_backend": video_backend,
            "music_backend": music_backend,
            "art_style": art_style,
            "scenes": max(1, min(scenes, 8)),
            "seconds_per_scene": max(2.0, min(seconds_per_scene, 15.0)),
            "music": music,
            "narrate": narrate,
            "seed": seed,
            "images": saved,
            "is_song": is_song,
        },
    )
    if is_song and not re.search(r"\b(song|sing|singing|rhyme|music)\b", prompt, re.I):
        # The storyboard keys song mode off these words, so make it explicit
        # without doubling the phrase if the user already wrote one.
        job.request["prompt"] = f"a song about {prompt}"

    JOBS[job_id] = job
    QUEUE.put(job_id)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="no such job")
    return job.public()


@app.post("/api/jobs/{job_id}/social")
def make_social(job_id: str, body: dict) -> dict:
    """Cut the finished video for each requested platform."""
    job = JOBS.get(job_id)
    if not job or job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="the video is not ready yet")

    keys = body.get("platforms") or list(PLATFORMS)
    source = WORK_DIR / job_id / Path(job.result["video"]).name
    board = WORK_DIR / job_id / "storyboard.json"

    try:
        manifest = export_all(
            source, board, WORK_DIR / job_id / "social",
            platforms=keys,
            style=body.get("style", "blur"),
            captions=bool(body.get("captions", True)),
        )
    except PipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    for entry in manifest:
        entry["file"] = f"/media/{job_id}/social/{Path(entry['file']).name}"
        entry["srt"] = f"/media/{job_id}/social/captions.srt"
    with _LOCK:
        job.social = manifest
    return {"social": manifest}


@app.get("/media/{job_id}/{name}")
def media(job_id: str, name: str, download: int = 0):
    path = (WORK_DIR / job_id / name).resolve()
    root = (WORK_DIR / job_id).resolve()
    if not str(path).startswith(str(root)) or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        path,
        media_type="video/mp4" if path.suffix == ".mp4" else None,
        filename=path.name if download else None,
    )


@app.get("/health")
def health() -> dict:
    import shutil as _sh

    return {
        "ok": True,
        "ffmpeg": bool(_sh.which("ffmpeg")),
        "url": "http://localhost:12000",
    }
