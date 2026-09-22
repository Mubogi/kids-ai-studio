#!/usr/bin/env python3
"""Publish the Kaggle notebook via the Kaggle API.

Kaggle has no "connect a repo" button like Vercel. A notebook is published by
pushing it with the API, and Kaggle copies the code into the user's own
account. The notebook then runs its own `git clone` of this repo for the
library code, so the two stay in sync automatically.

Credentials - either style works, the CLI's own resolution order is:

  1. Access token     KAGGLE_API_TOKEN env var, or ~/.kaggle/access_token
  2. Legacy API key   KAGGLE_USERNAME + KAGGLE_KEY env vars, or
                      ~/.kaggle/kaggle.json
  3. OAuth            `kaggle auth login`

The access token is the current style (kaggle.com -> Settings -> API).
The legacy username/key pair still works.

    python3 tools/publish_kaggle.py --repo-url https://github.com/USER/REPO

The notebook can also be created by hand:
Kaggle -> Create -> Notebook -> File -> Import Notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "kidvid_kaggle.ipynb"
LEGACY_CREDS = Path.home() / ".kaggle" / "kaggle.json"
LEGACY_DIR = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def detect_credentials() -> tuple[str, str | None]:
    """Return (method_description, kaggle_username_or_None).

    Mirrors the Kaggle CLI's own order so what we accept is what it accepts.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "access token from $KAGGLE_API_TOKEN", None

    token_file = LEGACY_DIR / "access_token"
    if token_file.exists():
        return f"access token from {token_file}", None

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "legacy key from $KAGGLE_USERNAME/$KAGGLE_KEY", os.environ["KAGGLE_USERNAME"]

    if LEGACY_CREDS.exists():
        try:
            data = json.loads(LEGACY_CREDS.read_text())
        except json.JSONDecodeError:
            die(f"{LEGACY_CREDS} is not valid JSON")
        if data.get("username") and data.get("key"):
            return f"legacy key from {LEGACY_CREDS}", data["username"]
        die(f"{LEGACY_CREDS} is missing 'username' or 'key'")

    die(
        "no Kaggle credentials found. Pick one:\n"
        "\n"
        "  Access token (current style):\n"
        "    1. https://www.kaggle.com/settings -> API -> Generate New Token\n"
        "    2. export KAGGLE_API_TOKEN=<the kgat_... token>\n"
        "\n"
        "  Legacy username/key (older style, still works):\n"
        "    1. https://www.kaggle.com/settings -> API -> Create New Token\n"
        "       (downloads kaggle.json)\n"
        f"    2. mkdir -p {LEGACY_CREDS.parent} && mv ~/Downloads/kaggle.json {LEGACY_CREDS}\n"
        f"       chmod 600 {LEGACY_CREDS}\n"
    )


def refresh_username(explicit: str | None, cli: list[str]) -> str:
    """Ask Kaggle who we are, so --username is optional."""
    if explicit:
        return explicit
    proc = subprocess.run([*cli, "config", "view"], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if "username" in line.lower():
            value = line.split(":", 1)[-1].strip()
            if value and value != "-":
                return value
    die("could not determine your Kaggle username; pass --username")


def resolve_cli() -> list[str]:
    """Return an argv prefix that runs the Kaggle CLI.

    The console script often lands in ~/.local/bin, which is not always on
    PATH here, so prefer whichever invocation actually works.
    """
    if shutil.which("kaggle"):
        return ["kaggle"]
    # `python -m kaggle` uses the same entry point and does not need PATH.
    probe = subprocess.run([sys.executable, "-m", "kaggle", "--version"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return [sys.executable, "-m", "kaggle"]
    return []


def ensure_cli() -> list[str]:
    cli = resolve_cli()
    if cli:
        return cli
    print("installing kaggle CLI ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    cli = resolve_cli()
    if not cli:
        die("could not run the kaggle CLI; try: pip install kaggle")
    return cli


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
    p.add_argument("--username", help="Kaggle username (auto-detected if omitted)")
    p.add_argument("--repo-url", required=True, help="public git URL of this repo")
    p.add_argument("--slug", default="kids-ai-studio",
                   help="notebook slug (URL name) on Kaggle")
    p.add_argument("--title", default="Kids AI Studio")
    p.add_argument("--dry-run", action="store_true",
                   help="build the upload folder but skip the actual push")
    p.add_argument("--public", action="store_true",
                   help="publish the notebook publicly (see note below)")
    args = p.parse_args()

    method, user_hint = detect_credentials()
    print(f"credentials: {method}")
    cli = ensure_cli()
    username = args.username or user_hint or refresh_username(None, cli)
    print(f"kaggle user: {username}")

    work = ROOT / "output" / "kaggle_upload"
    work.mkdir(parents=True, exist_ok=True)
    inject_repo_url(work / "notebook.ipynb", args.repo_url)

    # Kaggle returns 403 on saving a PUBLIC notebook over the API, so default
    # to private. Flip the visibility in the notebook's Settings menu on the
    # web UI if you want it public.
    (work / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{args.slug}",
        "title": args.title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": not args.public,
        # Free GPU. Kaggle bills this against the user's weekly GPU quota.
        "enable_gpu": True,
        "enable_internet": True,
        "accelerator": "nvidiaTeslaT4",
    }, indent=2))

    if args.dry_run:
        print(f"dry run: prepared {work} (not pushed)")
        return 0

    print("pushing to Kaggle ...")
    proc = subprocess.run([*cli, "kernels", "push", "-p", str(work)],
                          capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        die("kaggle push failed (is your API token still valid?)")

    print(f"\nDone. Notebook: https://www.kaggle.com/code/{username}/{args.slug}")
    print("Open it, then Run All. It will clone your repo and generate a video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
