"""Brand theme - Jordan Design Hub (JD Hub) yellow.

One palette used by every surface this project renders: the placeholder video
cards, the title card, the song captions, and the browser player. Change the
values here and the whole look follows.
"""

from __future__ import annotations

from dataclasses import dataclass

BRAND = "Jordan Design Hub (JD Hub)"
BRAND_LONG = "Mubogi Gastavas Jordan Tech Ecosystem"
CONTACT = "jordandesignhub@gmail.com"
WHATSAPP = "+256 754 687 597"


@dataclass(frozen=True)
class Theme:
    name: str
    primary: str    # main brand yellow
    deep: str       # darker amber, for contrast and borders
    soft: str       # pale yellow, backgrounds
    ink: str        # near-black warm ink, for text on yellow
    paper: str      # off-white
    accent: str     # secondary highlight

    # Scene gradient stops, one per storyboard beat.
    beats: dict[str, str]


JD_YELLOW = Theme(
    name="JD Hub Yellow",
    primary="#FFC107",
    deep="#E6A700",
    soft="#FFF4CE",
    ink="#2B2117",
    paper="#FFFBF0",
    accent="#FF8F00",
    beats={
        "setup": "#FFE082",
        "journey": "#FFD54F",
        "problem": "#FFB300",
        "resolution": "#FFCA28",
    },
)

DEFAULT_THEME = JD_YELLOW


def ffmpeg_color(hex_color: str, alpha: float | None = None) -> str:
    """Convert '#RRGGBB' into ffmpeg's '0xRRGGBB' form, optionally with alpha.

    ffmpeg drawtext/boxcolor want 0x-prefixed hex, and '@alpha' for opacity.
    """
    value = "0x" + hex_color.lstrip("#").upper()
    if alpha is None:
        return value
    return f"{value}@{alpha:.2f}"


def attribution() -> str:
    """Brand line burned into the title card."""
    return f"{BRAND}"
