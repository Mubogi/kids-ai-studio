#!/usr/bin/env python3
"""Generate ONE real video clip with Wan 2.2 on a CUDA GPU, then exit.

Runs inside a Kaggle notebook on a T4. Unlike the previous version of this
file, this drives the actual diffusers pipelines by name instead of guessing
at WanGP entrypoints, so failures are specific rather than silent.

Text-to-video:
    python tools/wangp_bridge.py --prompt "a happy turtle in a meadow" \
        --out /kaggle/working/clip.mp4

Image-to-video (animates a supplied first frame):
    python tools/wangp_bridge.py --prompt "..." --image start.png --out clip.mp4

Memory: the 5B transformer is ~10GB fp16 and the umt5-xxl text encoder is
~9GB, so the model cannot sit in a T4's 15.6GB all at once. We enable
model-level CPU offload, which moves one component onto the GPU at a time.
That works but adds host-device transfer on every step, so it is slower than
a 24GB card would be.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The weights are far larger than the default HF cache should be asked to
# hold in /kaggle/working, which is capped around 20GB.
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")

TI2V_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _log(msg: str) -> None:
    print(f"[bridge] {msg}", flush=True)


def _num_frames(seconds: float, fps: int) -> int:
    """Wan's VAE compresses 4 frames at a time, so the count must be 4n+1.

    Passing e.g. 60 gives a ragged last latent chunk and can either error or
    produce a subtly shorter clip, so snap down to the nearest 4n+1.
    """
    target = max(int(round(seconds * fps)), 5)
    frames = ((target - 1) // 4) * 4 + 1
    return max(frames, 5)


def _dims(width: int, height: int) -> tuple[int, int]:
    """Round to multiples of 32; the VAE downsamples by 16 and the patch
    embedder by 2, so odd sizes produce shape errors deep in the model."""
    w = max((width // 32) * 32, 128)
    h = max((height // 32) * 32, 128)
    return w, h


def _torch_dtype(name: str):
    import torch

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[name]

    # A T4 is sm_75, so bfloat16 is unsupported. Catch it here rather than
    # letting the first matmul fail with an opaque device-side assert.
    if name == "bf16" and torch.cuda.is_available() \
            and not torch.cuda.is_bf16_supported():
        _log("bfloat16 unsupported on this GPU; falling back to float16")
        return torch.float16
    return dtype


def build_pipeline(model: str, dtype: str, offload: bool):
    from diffusers import WanPipeline

    torch_dtype = _torch_dtype(dtype)
    _log(f"loading {model} as {torch_dtype}")
    pipe = WanPipeline.from_pretrained(
        model, torch_dtype=torch_dtype, use_safetensors=True)

    if offload:
        _log("enabling model CPU offload (weights do not fit VRAM together)")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # Tiling keeps the VAE decode from spiking on long or large clips.
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001 - optional optimisation
        pass
    return pipe


def build_i2v_pipeline(model: str, dtype: str, offload: bool):
    from diffusers import WanImageToVideoPipeline

    torch_dtype = _torch_dtype(dtype)
    _log(f"loading {model} as {torch_dtype}")
    pipe = WanImageToVideoPipeline.from_pretrained(
        model, torch_dtype=torch_dtype, use_safetensors=True)

    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001
        pass
    return pipe


def export(frames, out_path: Path, fps: int) -> None:
    """Write the generated frames to MP4.

    export_to_video accepts a list of HWC uint8 arrays or PIL images, and the
    exact return type of a pipeline has moved between diffusers releases, so
    let the helper normalise rather than assuming a shape here.
    """
    from diffusers.utils import export_to_video

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=fps)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=TI2V_REPO)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default=None)
    p.add_argument("--image", default=None, help="first frame for image-to-video")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--width", default=832, type=int)
    p.add_argument("--height", default=480, type=int)
    p.add_argument("--fps", default=24, type=int)
    p.add_argument("--seconds", default=5.0, type=float)
    p.add_argument("--steps", default=30, type=int)
    p.add_argument("--guidance", default=5.0, type=float)
    p.add_argument("--seed", default=1234, type=int)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--no-offload", action="store_true",
                   help="keep the whole model on GPU (needs >24GB)")
    args = p.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        _log("ERROR: no CUDA device. Set Accelerator -> GPU T4 x2 first.")
        return 2

    width, height = _dims(args.width, args.height)
    frames_n = _num_frames(args.seconds, args.fps)
    _log(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    _log(f"{width}x{height}, {frames_n} frames @ {args.fps}fps, "
         f"{args.steps} steps, guidance {args.guidance}")

    offload = not args.no_offload
    started = time.time()

    kwargs = dict(width=width, height=height, num_frames=frames_n,
                  num_inference_steps=args.steps,
                  guidance_scale=args.guidance,
                  generator=torch.Generator(device="cpu").manual_seed(args.seed))
    if args.negative:
        kwargs["negative_prompt"] = args.negative

    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        pipe = build_i2v_pipeline(args.model, args.dtype, offload)
        result = pipe(prompt=args.prompt, image=image, **kwargs)
    else:
        pipe = build_pipeline(args.model, args.dtype, offload)
        result = pipe(prompt=args.prompt, **kwargs)

    _log("generating (this is the slow part)")
    frames = result.frames[0]

    export(frames, args.out, args.fps)
    took = time.time() - started

    if not args.out.exists():
        _log("ERROR: pipeline returned but no file was written")
        return 1

    size_mb = args.out.stat().st_size / 1e6
    _log(f"wrote {args.out} ({size_mb:.1f} MB) in {took / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

