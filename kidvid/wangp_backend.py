"""Adapter that runs real open models on a free Kaggle GPU.

Honest note on how this works
-----------------------------
WanGP's public interface is a Gradio web app plus a CLI whose flags change
between releases. Rather than hard-code flags that may not exist, this
backend shells out to `tools/wangp_bridge.py`, which you run inside the
Kaggle notebook where WanGP is installed. If the bridge cannot find a
working entrypoint it stops with an explicit message instead of guessing.

That keeps this package runnable on any machine (MockVideoBackend) while
the GPU-heavy part stays in the notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import ShowConfig
from .storyboard import Scene, negative_for
from .util import run, PipelineError


class WanGPBackend:
    """Generate one clip per scene using Wan 2.2 / LTX inside WanGP."""

    def __init__(
        self,
        model: str = "wan2.2_ti2v_5B_Q4_K_M",
        bridge: str = "tools/wangp_bridge.py",
        python: str | None = None,
    ) -> None:
        """Defaults to the Q4_K_M quantised Wan 2.2 checkpoint, which is the
        variant that fits Kaggle's 16GB T4. On a 24GB+ card, pass
        ``model="wan2.2_ti2v_5B"`` for bf16 quality."""
        self.model = model
        self.bridge = Path(bridge)
        self.python = python or sys.executable

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
        ]
        # Image-to-video when the caller supplied a starting picture.
        if scene.image:
            cmd += ["--image", str(scene.image)]

        run(cmd, quiet=False)
        if not out_path.exists():
            raise PipelineError(f"backend reported success but {out_path} is missing")
        return out_path
