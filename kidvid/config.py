from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class ShowConfig:
    """Everything needed to turn one prompt into one finished video."""

    prompt: str
    title: str = "My Story"
    age_range: str = "3-6"

    # structure
    scene_count: int = 4
    seconds_per_scene: float = 5.0

    # video
    width: int = 832
    height: int = 480
    fps: int = 24
    steps: int = 30
    guidance: float = 5.0
    # Each scene's first frame comes from one of these images when provided.
    image_paths: list[str] = field(default_factory=list)

    # music / song
    make_music: bool = True
    music_style: str = "cheerful children's song, ukulele, glockenspiel, whistling, upbeat"
    music_seconds: float = 20.0

    # narration
    narrate: bool = False
    voice: str = "en+f3"
    words_per_second: float = 2.4

    # output
    out_dir: str = "output"
    seed: int = 1234

    def scene_images(self) -> list[str]:
        return list(self.image_paths)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "ShowConfig":
        return cls(**json.loads(Path(path).read_text()))
