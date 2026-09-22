#!/usr/bin/env python3
"""Publish the Kaggle notebook via the Kaggle API.

Kaggle has no "connect a repo" button like Vercel. A notebook is published by
pushing it with the API, and Kaggle copies the code into the user's own
account. The notebook then runs its own `git clone` of this repo for the
library code, so the two stay in sync automatically.

Requires credentials from kaggle.com -> Settings -> API -> Create New Token,
saved as ~/.kaggle/kaggle.json (chmod 600).

    python3 tools/publish_kaggle.py --username YOUR_KAGGLE_USER --repo-url https://github.com/USER/REPO

This is a convenience wrapper. The notebook can also be created by hand:
Kaggle -> Create -> Notebook -> File -> Import Notebook.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "kidvid_kaggle.ipynb"
CREDS = Path.home() / ".kaggle" / "kaggle.json"


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def ensure_credentials() -> None:
    if not CREDS.exists():
        die(
            f"no Kaggle credentials at {CREDS}\n"
            "  1. Go to https://www.kaggle.com/settings\n"
            "  2. API -> Create New Token (downloads kaggle.json)\n"
            f"  3. Move it to {CREDS} and run: chmod 600 {CREDS}"
        )
    try:
        data = json.loads(CREDS.read_text())
    except json.JSONDecodeError:
        die(f"{CREDS} is not valid JSON")
    for key in ("username", "key"):
        if not data.get(key):
            die(f"{CREDS} is missing '{key}'")
    print(f"credentials OK for Kaggle user: {data['username']}")


def ensure_cli() -> None:
    if shutil.which("kaggle"):
        return
    print("installing kaggle CLI ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    if not shutil.which("kaggle"):
        die("kaggle CLI not on PATH after install; try: pip install kaggle")


def inject_repo_url(work: Path, repo_url: str) -> None:
    """Point the notebook's REPO_URL cell at this repo."""
    nb = json.loads(NOTEBOOK.read_text())
    patched = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if 'REPO_URL = "https://github.com/YOUR_USERNAME/kidvid.git"' in src:
            cell["source"] = [
                line.replace(
                    'REPO_URL = "https://github.com/YOUR_USERNAME/kidvid.git"',
                    f'REPO_URL = "{repo_url}"',
                )
                for line in cell["source"]
            ]
            patched += 1
    if not patched:
        die("could not find the REPO_URL placeholder in the notebook")
    work.write_text(json.dumps(nb, indent=1))
    print(f"injected REPO_URL = {repo_url}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", required=True, help="your Kaggle username")
    p.add_argument("--repo-url", required=True, help="public git URL of this repo")
    p.add_argument("--slug", default="kids-ai-studio",
                   help="notebook slug (URL name) on Kaggle")
    p.add_argument("--title", default="Kids AI Studio")
    p.add_argument("--dry-run", action="store_true",
                   help="build the upload folder but skip the actual push")
    args = p.parse_args()

    ensure_credentials()
    ensure_cli()

    work = ROOT / "output" / "kaggle_upload"
    work.mkdir(parents=True, exist_ok=True)
    inject_repo_url(work / "notebook.ipynb", args.repo_url)

    (work / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.username}/{args.slug}",
        "title": args.title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        # Free GPU. Kaggle will bill against the user's weekly GPU quota.
        "enable_gpu": True,
        "enable_internet": True,
        # Free-tier accelerators
        "accelerator": "nvidiaTeslaT4",
    }, indent=2))

    if args.dry_run:
        print(f"dry run: prepared {work} (not pushed)")
        return 0

    print("pushing to Kaggle ...")
    proc = subprocess.run(["kaggle", "kernels", "push", "-p", str(work)],
                          capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        die("kaggle push failed (is your API token still valid?)")

    print(f"\nDone. Notebook: https://www.kaggle.com/code/{args.username}/{args.slug}")
    print("Open it, then Run All. It will clone your repo and generate a video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
