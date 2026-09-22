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
# Recommended by the CUDA OOM message itself. The T4 run failed while only
# 160MB short, which is the signature of allocator fragmentation rather than
# genuinely needing more memory, and this reuses fragmented blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def _apply_offload(pipe, mode: str) -> None:
    """Place the pipeline on the GPU as frugally as the mode allows.

    `model` moves one whole component at a time and is the normal choice,
    but on a 15.6GB T4 the 5B transformer (~10GB) and the umt5-xxl text
    encoder (~9GB) do not fit together, and even with offload the hand-off
    between them ran out of memory on a real run.

    `balanced` is handled earlier, in from_pretrained, because it needs
    accelerate's dispatch hooks rather than post-hoc placement.

    `sequential` is supported but not recommended on Kaggle: it holds all
    weights in host RAM (~20GB), and a session only has ~13GB, so the kernel
    gets OOM-killed during loading.
    """
    if mode == "none":
        pipe.to("cuda")
    elif mode == "model":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise ValueError(f"unknown offload mode: {mode}")

    # Both help the VAE decode, which is where video pipelines spike: the
    # whole clip is decoded in one call, unlike the single frame of an
    # image model.
    for fn in ("enable_tiling", "enable_slicing"):
        try:
            getattr(pipe.vae, fn)()
        except Exception:  # noqa: BLE001 - optional optimisations
            pass


def load_pipeline(kind: str, model: str, dtype: str, offload: str):
    """Build the text-to-video ("t2v") or image-to-video ("i2v") pipeline."""
    from diffusers import WanImageToVideoPipeline, WanPipeline

    cls = WanImageToVideoPipeline if kind == "i2v" else WanPipeline
    torch_dtype = _torch_dtype(dtype)
    _log(f"loading {model} as {torch_dtype} (offload={offload})")

    kwargs = dict(torch_dtype=torch_dtype, use_safetensors=True,
                  low_cpu_mem_usage=True)

    if offload == "balanced":
        # accelerate shards the components across the visible GPUs and
        # inserts dispatch hooks that move activations to whichever device
        # holds the module being run. Doing this by hand with .to() would
        # leave prompt embeddings on the text encoder's card while the
        # latents sat on the transformer's, which faults.
        #
        # device_map="balanced" fills each card to the brim, and that fails
        # on a T4: cuBLAS needs its own scratch space for a GEMM handle, and
        # with no room left cublasCreate returns ALLOC_FAILED mid-generation.
        # max_memory reserves headroom so the library still has somewhere to
        # put its workspace.
        headroom = float(os.environ.get("WAN_MAX_MEM_HEADROOM_GB", "2"))
        import torch

        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            gib = max(int(total / 2 ** 30 - headroom), 2)
            max_memory[i] = f"{gib}GiB"
            _log(f"  cuda:{i} budget {gib}GiB (of "
                 f"{total / 2 ** 30:.1f}GiB, {headroom:.0f}GiB headroom)")
        kwargs["device_map"] = "balanced"
        kwargs["max_memory"] = max_memory
        pipe = cls.from_pretrained(model, **kwargs)
        # tiling/slicing still apply; no offload call, accelerate owns placement.
        for fn in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, fn)()
            except Exception:  # noqa: BLE001
                pass
        return pipe

    pipe = cls.from_pretrained(model, **kwargs)
    _apply_offload(pipe, offload)
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
    p.add_argument("--offload", default="auto",
                   choices=["auto", "balanced", "model", "sequential", "none"],
                   help="auto: balanced across GPUs if there are 2+, else "
                        "model offload (default); balanced: device_map across "
                        "all GPUs; model: one component on GPU at a time; "
                        "sequential: every submodule, needs lots of host RAM; "
                        "none: whole model in VRAM, needs >24GB")
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

    # Retry order on CUDA OOM. `balanced` first when several GPUs are
    # present: it spreads components over real VRAM rather than shuffling
    # them to and from host memory. `sequential` is excluded because it was
    # tried on a real T4 and the kernel was OOM-killed at 65% of weight
    # loading - it keeps every weight in host RAM (~20GB) and a Kaggle
    # session has ~13GB.
    n_gpus = torch.cuda.device_count()
    modes = []
    if args.offload == "none":
        modes = ["none"]
    elif args.offload in ("model", "sequential"):
        modes = [args.offload]
    elif n_gpus >= 2:
        # `auto` and explicit `balanced` both want the multi-GPU path first,
        # with single-card offload as the fallback.
        modes = ["balanced", "model"]
    else:
        modes = ["model"]

    _log(f"{n_gpus} GPU(s) visible; will try: {', '.join(modes)}")

    kind = "i2v" if args.image else "t2v"
    image = None
    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")

    started = time.time()
    result = None
    pipe = None
    # One attempt per mode, then shrinking resolutions for the last mode.
    attempts: list[tuple[str, float]] = [(m, 1.0) for m in modes]
    attempts += [(modes[-1], 0.5), (modes[-1], 0.25)]

    for attempt, (mode, scale) in enumerate(attempts):
        # Shrink on the later attempts. Peak VRAM tracks the latent volume,
        # so cutting the frame count helps far more than trimming width.
        cur_frames = max(_num_frames(args.seconds * scale, args.fps), 5)
        cur_w = _dims(max(int(width * scale), 256), max(int(height * scale), 256))
        if attempt:
            _log(f"attempt {attempt}: offload={mode}, "
                 f"{cur_w[0]}x{cur_w[1]}, {cur_frames} frames")

        kwargs = dict(width=cur_w[0], height=cur_w[1], num_frames=cur_frames,
                      num_inference_steps=args.steps,
                      guidance_scale=args.guidance,
                      generator=torch.Generator(device="cpu").manual_seed(args.seed))
        if args.negative:
            kwargs["negative_prompt"] = args.negative

        try:
            pipe = load_pipeline(kind, args.model, args.dtype, mode)
            _log("generating (this is the slow part)")
            if image is not None:
                result = pipe(prompt=args.prompt, image=image, **kwargs)
            else:
                result = pipe(prompt=args.prompt, **kwargs)
            break
        except torch.OutOfMemoryError:
            result, pipe = None, None
            torch.cuda.empty_cache()
            if attempt + 1 == len(attempts):
                _log("ERROR: out of memory at every setting tried. "
                     "Try --height 384 --seconds 1.")
                return 3
            _log("out of memory; trying the next configuration")

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

