"""Launch the JD Hub Kids AI Studio web interface.

    python3 -m kidvid.serve            # http://localhost:12000
    python3 -m kidvid.serve --port 8000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kidvid.serve",
                                description="Run the Kids AI Studio interface.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=12000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    import uvicorn

    print(f"JD Hub Kids AI Studio -> http://localhost:{args.port}")
    uvicorn.run("kidvid.app:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
