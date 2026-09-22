"""Generate a branded, standalone HTML preview page for a finished video.

Run: python3 tools/build_preview.py output/final_titled.mp4
Writes preview.html next to the video. No server, no build step - open it
and the video, storyboard, and brand card are all there.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kidvid.theme import BRAND, BRAND_LONG, CONTACT, DEFAULT_THEME, WHATSAPP


def build(video: Path, storyboard: Path | None = None) -> Path:
    theme = DEFAULT_THEME
    data = base64.b64encode(video.read_bytes()).decode("ascii")

    board = {}
    if storyboard and storyboard.exists():
        board = json.loads(storyboard.read_text())

    scenes_html = "\n".join(
        f"""      <article class="scene">
        <span class="beat">{s['beat']}</span>
        <h3>Scene {s['index'] + 1}</h3>
        <p class="line">{s['narration']}</p>
        <p class="prompt">{s['video_prompt']}</p>
      </article>"""
        for s in board.get("scenes", [])
    )

    song_html = ""
    if board.get("song"):
        lines = "\n".join(f"        <li>{line}</li>" for line in board["song"])
        song_html = f"""
    <section class="card">
      <h2>Lyrics</h2>
      <ul class="lyrics">
{lines}
      </ul>
    </section>"""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{board.get('title', 'Kids Video')} — {BRAND}</title>
<style>
  :root {{
    --primary: {theme.primary};
    --deep: {theme.deep};
    --soft: {theme.soft};
    --ink: {theme.ink};
    --paper: {theme.paper};
    --accent: {theme.accent};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--paper);
    color: var(--ink);
    line-height: 1.55;
  }}
  header {{
    background: linear-gradient(135deg, var(--primary), var(--deep));
    padding: 2rem 1.25rem;
    border-bottom: 6px solid var(--ink);
  }}
  header .wrap {{ max-width: 960px; margin: 0 auto; }}
  .brand {{
    display: inline-block;
    background: var(--ink);
    color: var(--primary);
    font-weight: 700;
    font-size: .72rem;
    letter-spacing: .13em;
    text-transform: uppercase;
    padding: .35rem .8rem;
    border-radius: 999px;
  }}
  header h1 {{
    margin: .7rem 0 .25rem;
    font-size: clamp(1.5rem, 4vw, 2.4rem);
    color: var(--ink);
  }}
  header p {{ margin: 0; font-weight: 500; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 1.5rem 1.25rem 4rem; }}
  video {{
    width: 100%;
    border-radius: 14px;
    border: 4px solid var(--ink);
    background: #000;
    box-shadow: 0 12px 30px rgba(43,33,23,.22);
  }}
  .card {{
    background: #fff;
    border: 2px solid var(--soft);
    border-left: 6px solid var(--primary);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-top: 1.5rem;
  }}
  h2 {{ margin: 0 0 .8rem; font-size: 1.1rem; }}
  .scenes {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
  .scene {{
    background: var(--soft);
    border-radius: 12px;
    padding: .9rem 1rem;
    border: 2px solid #fff;
  }}
  .scene h3 {{ margin: .35rem 0; font-size: .95rem; }}
  .beat {{
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    background: var(--primary);
    color: var(--ink);
    padding: .2rem .55rem;
    border-radius: 999px;
  }}
  .line {{ font-size: .88rem; margin: .3rem 0; }}
  .prompt {{ font-size: .74rem; color: #6b5c47; margin: .45rem 0 0; }}
  .lyrics {{ margin: 0; padding-left: 1.15rem; }}
  .lyrics li {{ margin-bottom: .3rem; }}
  footer {{
    background: var(--ink);
    color: var(--soft);
    padding: 1.5rem 1.25rem;
    font-size: .82rem;
    text-align: center;
  }}
  footer strong {{ color: var(--primary); }}
  footer a {{ color: var(--primary); }}
</style>
</head>
<body>
  <header>
    <div class="wrap">
      <span class="brand">{BRAND}</span>
      <h1>{board.get('title', 'Kids Video')}</h1>
      <p>{BRAND_LONG}</p>
    </div>
  </header>

  <main>
    <video controls playsinline src="data:video/mp4;base64,{data}"></video>

    <section class="card">
      <h2>Storyboard</h2>
      <div class="scenes">
{scenes_html}
      </div>
    </section>{song_html}
  </main>

  <footer>
    <p><strong>{BRAND}</strong> — {BRAND_LONG}</p>
    <p><a href="mailto:{CONTACT}">{CONTACT}</a> · WhatsApp {WHATSAPP}</p>
    <p>Generated with open-source models on a free GPU.</p>
  </footer>
</body>
</html>
"""
    out = video.parent / "preview.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: build_preview.py <video.mp4> [storyboard.json]")
    video_path = Path(sys.argv[1])
    board_path = Path(sys.argv[2]) if len(sys.argv) > 2 else video_path.parent / "storyboard.json"
    page = build(video_path, board_path)
    print(f"wrote {page} ({page.stat().st_size / 1e6:.1f} MB)")
