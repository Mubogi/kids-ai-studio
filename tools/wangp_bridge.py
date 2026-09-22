#!/usr/bin/env python3
"""Bridge: generate ONE clip with WanGP on a Kaggle GPU, then exit.

Run this only inside the Kaggle notebook after WanGP is installed:

    cd /kaggle/working/Wan2GP
    python tools/wangp_bridge.py --model wan2.2_ti2v_5B \
        --prompt "a happy turtle in a meadow" --out /kaggle/working/clip.mp4

WanGP is driven through its Python entrypoints. The names below are the ones
WanGP has exposed historically; `--list` prints what this install supports.
If nothing matches, run the WanGP Gradio UI and use MockVideoBackend to
finish the pipeline - do not hack around it, the flags move between releases.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path


def _candidate_entrypoints():
    """Importable (module, attr) pairs that may expose WanGP generation."""
    return [
        ("wan", "generate"),
        ("wgp", "generate"),
        ("source.generate", "generate"),
        ("source.wan_handler", "generate"),
        ("shared.gradio_ui", "generate"),
    ]


def list_entrypoints() -> int:
    print("Scanning for WanGP entrypoints on this machine:\n")
    found = 0
    for module, attr in _candidate_entrypoints():
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report and keep scanning
            print(f"  [ ] {module}.{attr:14} not importable ({type(exc).__name__})")
            continue
        fn = getattr(mod, attr, None)
        if fn is None:
            print(f"  [ ] {module}.{attr:14} module ok, attribute missing")
            continue
        found += 1
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(signature unavailable)"
        print(f"  [x] {module}.{attr}{sig}")

    if not found:
        print(
            "\nNo entrypoint found. Use the WanGP Gradio UI shown in the "
            "notebook and keep MockVideoBackend for pipeline testing."
        )
    return 0 if found else 1


def generate(args: argparse.Namespace) -> int:
    for module, attr in _candidate_entrypoints():
        try:
            mod = importlib.import_module(module)
        except Exception:
            continue
        fn = getattr(mod, attr, None)
        if fn is None:
            continue

        params = set(inspect.signature(fn).parameters)
        kwargs: dict = {}
        mapping = {
            "prompt": args.prompt,
            "negative_prompt": args.negative,
            "save_path": str(args.out),
            "output_path": str(args.out),
            "model_type": args.model,
            "resolution": f"{args.width}x{args.height}",
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
            "video_length": int(args.seconds * args.fps),
            "image_start": args.image,
        }
        for key, value in mapping.items():
            if key in params and value is not None:
                kwargs[key] = value

        print(f"using {module}.{attr} with kwargs={sorted(kwargs)}", file=sys.stderr)
        fn(**kwargs)
        return 0 if Path(args.out).exists() else 1

    print(
        "ERROR: no usable WanGP entrypoint found.\n"
        "Run `python tools/wangp_bridge.py --list` on the GPU machine, then "
        "either match its signature here or generate via the WanGP web UI.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="show available entrypoints")
    p.add_argument("--model", default="wan2.2_ti2v_5B")
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative", default=None)
    p.add_argument("--image", default=None, help="first frame for image-to-video")
    p.add_argument("--out", default=None, type=Path)
    p.add_argument("--width", default=832, type=int)
    p.add_argument("--height", default=480, type=int)
    p.add_argument("--fps", default=24, type=int)
    p.add_argument("--seconds", default=5.0, type=float)
    p.add_argument("--steps", default=30, type=int)
    p.add_argument("--guidance", default=5.0, type=float)
    p.add_argument("--seed", default=1234, type=int)
    args = p.parse_args()

    if args.list:
        return list_entrypoints()
    if not args.prompt or not args.out:
        p.error("--prompt and --out are required unless --list is used")
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
