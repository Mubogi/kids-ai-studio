"""Adapter that runs real open video models on a free Kaggle GPU.

This is the only backend here that produces genuine frame-by-frame motion.
The image backends in ai_backends.py animate a still with a camera move;
this one runs Wan 2.2 diffusion to synthesise actual movement.

It shells out to `tools/wangp_bridge.py`, which calls the diffusers Wan
pipelines directly. The GPU-heavy part therefore runs on the Kaggle box
while this package stays importable on a laptop.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import ShowConfig
from .storyboard import Scene, negative_for
from .util import run, PipelineError

# The diffusers repo for the 5B hybrid text/image-to-video model. Note this
# is the *transformers* checkpoint, not a WanGP GGUF name - the previous
# default here was a WanGP-style quantisation tag that the diffusers
# pipeline would not have recognised.
DEFAULT_WAN_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class WanGPBackend:
    """Generate one clip per scene with Wan 2.2 on CUDA."""

    def __init__(
        self,
        model: str = DEFAULT_WAN_MODEL,
        bridge: str = "tools/wangp_bridge.py",
        python: str | None = None,
        dtype: str = "fp16",
        offload: str = "auto",
    ) -> None:
        """Defaults suit a 16GB T4: fp16, picking the offload mode at runtime.

        A T4 is sm_75 and has no bfloat16, so fp16 is the right precision.
        ``offload="auto"`` uses accelerate's balanced device map when the
        machine has two or more GPUs, and falls back to single-card model
        offload otherwise; the bridge also retries at lower resolutions if it
        runs out of memory either way. On a 24GB+ card, ``offload="none"``
        avoids the per-step host-device copies.
        """
        self.model = model
        self.bridge = Path(bridge)
        self.python = python or sys.executable
        self.dtype = dtype
        self.offload = offload

    def generate(self, scene: Scene, cfg: ShowConfig, out_path: Path) -> Path:
        if not self.bridge.exists():
            raise PipelineError(
                f"bridge script not found: {self.bridge}. "
                "Run this on the Kaggle GPU machine where the repo is cloned."
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python, str(self.bridge),
            "--model", self.model,
            "--prompt", scene.video_prompt,
            "--negative", negative_for(cfg.art_style),
            "--out", str(out_path),
            "--width", str(cfg.width),
            "--height", str(cfg.height),
            "--fps", str(cfg.fps),
            "--seconds", f"{scene.seconds:.2f}",
            "--steps", str(cfg.steps),
            "--guidance", str(cfg.guidance),
            "--seed", str(cfg.seed + scene.index),
            "--dtype", self.dtype,
            "--offload", self.offload,
        ]
        # Image-to-video when the caller supplied a starting picture.
        if scene.image:
            cmd += ["--image", str(scene.image)]

        run(cmd, quiet=False)
        if not out_path.exists():
            raise PipelineError(f"backend reported success but {out_path} is missing")
        return out_path

