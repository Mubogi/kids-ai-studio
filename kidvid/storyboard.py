"""Turn one plain-English idea into a scene-by-scene storyboard for kids.

Deliberately dependency-free and deterministic: the same prompt always
produces the same storyboard. Swap in an LLM later via `storyboard_fn`
if you want richer writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .config import ShowConfig

# Appended to every scene prompt so the models stay in a kid-safe register.
SAFE_STYLE = (
    "colorful 2D animated cartoon for young children, cute friendly characters, "
    "bright soft pastel colors, gentle warm lighting, rounded simple shapes, "
    "storybook illustration style, no text, no words, no letters, wholesome and cheerful"
)

NEGATIVE_PROMPT = (
    "scary, frightening, violent, blood, gore, weapons, dark horror lighting, "
    "sad crying close-up, realistic human faces, extra limbs, deformed hands, "
    "text, watermark, logo, signature, low quality, blurry, jittery"
)

# Beat names describe the shape of a classic picture-book arc.
_BEATS = ("setup", "journey", "problem", "resolution")

_FILLER = {
    "a", "an", "the", "little", "big", "small", "brave", "happy", "sad",
    "who", "that", "which", "and", "with", "his", "her", "their", "learns",
    "learn", "to", "about", "story", "song", "video", "make", "of", "for",
}


def _strip_lead_in(prompt: str) -> str:
    """Remove framing words so the character is found, not the framing.

    "a song about a friendly dragon" -> "a friendly dragon"
    """
    text = re.sub(r"\s+", " ", prompt.strip().rstrip(".!?"))
    pattern = (
        r"^(?:please\s+)?(?:make|create|generate|do|write|i\s+want|i'd\s+like|"
        r"can\s+you\s+make)?\s*"
        r"(?:a|an|the)?\s*(?:song|story|video|movie|cartoon|animation|"
        r"nursery\s+rhyme|rhyme|tale)\s+"
        r"(?:about|of|for|with|featuring)?\s*"
    )
    stripped = re.sub(pattern, "", text, flags=re.I).strip()
    return stripped or text


# Verbs a child's prompt is likely to start with. Used only to find where the
# character ends and the action begins; the fallback below covers anything else.
_VERBS = {
    "goes", "go", "went", "learns", "learn", "learnt", "finds", "find", "found",
    "wants", "want", "makes", "make", "made", "plays", "play", "sings", "sing",
    "dances", "dance", "flies", "fly", "swims", "swim", "runs", "run", "helps",
    "help", "shares", "share", "eats", "eat", "sleeps", "sleep", "jumps", "jump",
    "climbs", "climb", "explores", "explore", "travels", "travel", "visits",
    "visit", "meets", "meet", "saves", "save", "builds", "build", "draws",
    "draw", "reads", "read", "counts", "count", "paints", "paint", "loves",
    "love", "hides", "hide", "seeks", "seek", "rides", "ride", "walks", "walk",
}


def _split_prompt(prompt: str) -> tuple[str, str, str]:
    """Split a prompt into (article, character, theme).

    "a brave little turtle who learns to share"
        -> ("a", "brave little turtle", "learns to share")
    The theme is the *action*, not the whole sentence, so narrations and
    music prompts don't just repeat the character back at the child.
    """
    text = _strip_lead_in(prompt)
    if not text:
        return "a", "little friend", "a happy adventure"

    # "a <character> who <action>" - the most useful shape for kids' prompts.
    match = re.search(
        r"^\s*(a|an|the)\s+(.+?)\s+(?:who|that|which)\s+(.+)$", text, re.I
    )
    if match:
        return match.group(1).lower(), match.group(2).strip().lower(), match.group(3).strip()

    # "<character> named <name> ..."
    match = re.search(r"^\s*(a|an|the)?\s*(.+?)\s+(?:named|called)\s+(.+)$", text, re.I)
    if match:
        article = (match.group(1) or "a").lower()
        return article, match.group(2).strip().lower(), match.group(3).strip()

    words = re.findall(r"[A-Za-z']+", text)
    article = "a"
    if words and words[0].lower() in ("a", "an", "the"):
        article = words.pop(0).lower()

    # "my dog", "our cat" - the child says "my", narration should not.
    if words and words[0].lower() in ("my", "our", "his", "her", "their"):
        words.pop(0)
        article = "the"

    if not words:
        return article, "little friend", "a happy adventure"

    cut = _verb_index(words)
    character = " ".join(words[:cut]).lower()
    theme = " ".join(words[cut:]).lower() or f"a happy day with the {character}"
    return article, character, theme


def _verb_index(words: list[str]) -> int:
    """Index of the action verb, or len(words) when there isn't one."""
    for i, word in enumerate(words):
        if i == 0:
            continue
        if word.lower() in _VERBS:
            return i
    return len(words)


def _article_for(character: str) -> str:
    """Pick a/an by sound; vowel-initial words need "an" except a few."""
    first = character.split()[0].lower()
    if first.startswith(("uni", "eu", "one", "use")):
        return "a"
    return "an" if first[:1] in "aeiou" else "a"


def _infinitive(theme: str) -> str:
    """Turn a theme clause into an infinitive phrase.

    "learns to share" -> "to learn to share"; "goes to space" -> "to go to space".
    Only strips third-person -s, which covers the prompts people actually type.
    """
    theme = theme.strip()
    if not theme:
        return "have a happy day"
    if theme.lower().startswith("to "):
        return theme

    first, _, rest = theme.partition(" ")
    low = first.lower()
    if low.endswith("ies") and len(low) > 4:
        first = first[:-3] + "y"
    elif low.endswith(("ches", "shes", "xes", "sses")) and len(low) > 4:
        first = first[:-2]
    elif low.endswith("oes") and len(low) > 3:
        first = first[:-2]          # goes -> go, does -> do
    elif low.endswith("s") and not low.endswith(("ss", "us", "is")) and len(low) > 3:
        first = first[:-1]          # learns -> learn, dances -> dance

    return f"to {first} {rest}".strip()


def _action(theme: str) -> str:
    """Infinitive without the "to", for sentences that already supply it."""
    infinitive = _infinitive(theme)
    return infinitive[3:] if infinitive.startswith("to ") else infinitive


def _narration(character: str, beat: str, theme: str) -> str:
    """Short, read-aloud-friendly narration for each beat."""
    name = character.split()[-1]
    return {
        "setup": f"Once upon a time, there was {_article_for(character)} {character}. "
                 f"Today, the {name} wanted {_infinitive(theme)}.",
        "journey": f"So the {name} set off, "
                   f"wondering what wonderful things were waiting to be found.",
        "problem": f"Oh no! The {name} ran into a bumpy, tricky problem. "
                   f"What could the {name} do now?",
        "resolution": f"The {name} had a kind idea, and it worked! "
                      f"Everyone cheered, and everyone felt warm and happy.",
    }[beat]


def _scene_prompt(character: str, beat: str, theme: str, camera: str) -> str:
    """A concrete shot description handed to the video model."""
    idea = _action(theme)
    shots = {
        "setup": f"Opening wide shot: {character} waking up happily in a sunny meadow "
                 f"full of flowers, birds flying, ready to {idea}",
        "journey": f"Tracking shot: {character} walking cheerfully through a bright "
                   f"colorful forest path, butterflies and friendly animals waving hello",
        "problem": f"Medium shot: {character} looking surprised and puzzled by a small "
                   f"friendly obstacle, gentle expression, soft colors, not scary",
        "resolution": f"Warm closing shot: {character} smiling with new friends, "
                      f"confetti and sparkles, golden sunset, everyone celebrating together",
    }
    return f"{shots[beat]}. {camera}. {SAFE_STYLE}"


def _camera_for(beat: str) -> str:
    return {
        "setup": "smooth slow push in, stable camera",
        "journey": "gentle sideways dolly, stable camera",
        "problem": "slow subtle zoom, stable camera",
        "resolution": "slow pull back revealing the scene, stable camera",
    }[beat]


# Simple rhyme banks so songs scan without an LLM. Repetition is deliberate:
# nursery rhymes work by repeating a chorus a child can join in with.
_RHYMES = [
    ("day", "play"), ("sun", "fun"), ("tree", "free"),
    ("song", "along"), ("sky", "high"), ("blue", "too"),
]


def _song_goal(theme: str) -> str:
    """A singable action phrase, without "to" (callers add it).

    Themes like "a day with friendly dragon" are noun phrases and read badly
    after "wanted to", so they fall back to a generic cheerful action.
    """
    if re.match(r"^(a|an|the|my|our|his|her|their)\b", theme.strip(), re.I):
        return "have a happy day"
    return _action(theme) or "have a happy day"


def write_song(character: str, theme: str, verses: int = 2) -> list[str]:
    """Build a short, repetitive, singable lyric set for young children."""
    name = character.split()[-1]
    goal = _song_goal(theme)

    chorus = [
        f"Oh {name}, {name}, what a {_RHYMES[0][0]}!",
        f"We sing and laugh and {_RHYMES[0][1]} along the way.",
    ]

    # Each verse opens with a different, self-contained couplet so nothing
    # depends on the surrounding grammar.
    verses_bank = [
        [f"The {name} wanted to {goal},",
         f"and found a happy way to do it!"],
        [f"The {name} skipped and hopped along,",
         f"and sang a happy song all day!"],
        [f"The {name} made a brand new friend,",
         f"and they played until the sun went down!"],
    ]

    lines: list[str] = []
    for verse in range(max(1, verses)):
        lines += verses_bank[verse % len(verses_bank)]
        lines += [
            f"So clap your hands and stomp your feet,",
            *chorus,
        ]
    lines += [
        f"So clap for {name}, one, two, three!",
        f"The happiest friends you ever did see!",
    ]
    return lines


@dataclass
class Scene:
    index: int
    beat: str
    narration: str
    video_prompt: str
    seconds: float
    image: str | None = None


@dataclass
class Storyboard:
    title: str
    theme: str
    character: str
    scenes: list[Scene] = field(default_factory=list)
    song: list[str] = field(default_factory=list)
    music_prompt: str = ""

    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.scenes)


def build_storyboard(cfg: ShowConfig) -> Storyboard:
    """Expand a ShowConfig into a full kid-safe storyboard."""
    article, character, theme = _split_prompt(cfg.prompt)

    images = cfg.scene_images()
    scenes: list[Scene] = []

    for i in range(cfg.scene_count):
        beat = _BEATS[i] if i < len(_BEATS) else _BEATS[-1]
        scenes.append(
            Scene(
                index=i,
                beat=beat,
                narration=_narration(character, beat, theme),
                video_prompt=_scene_prompt(character, beat, theme, _camera_for(beat)),
                seconds=cfg.seconds_per_scene,
                image=images[i] if i < len(images) else None,
            )
        )

    is_song = bool(re.search(r"\b(song|sing|singing|rhyme|music)\b", cfg.prompt, re.I))
    song = write_song(character, theme) if is_song else []

    subject = f"{article} {character}"
    return Storyboard(
        title=cfg.title,
        theme=f"{subject} who {theme}" if is_song else subject,
        character=character,
        scenes=scenes,
        song=song,
        music_prompt=f"{cfg.music_style}. About {subject}. Instrumental, no vocals.",
    )
