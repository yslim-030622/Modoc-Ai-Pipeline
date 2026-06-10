"""Campaign prompt profiles for meme planning.

Profiles keep creative direction out of the prompt-building code path. The
default remains conservative for pediatric education, but it avoids locking a
different character design per language.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CAMPAIGN_PROFILE: dict[str, Any] = {
    "name": "trend_first_parent_education",
    "trend_window": "last 30-60 days when current search results are available",
    "candidate_count_per_language": 3,
    "scene_count": 4,
    "languages": {
        "english": {
            "platforms": ["TikTok", "Instagram Reels", "YouTube Shorts"],
            "native_context": "US/global parent education short-form.",
            "language_note": "Use natural English parent phrasing. Avoid forcing Gen-Z slang.",
        },
        "korean": {
            "platforms": ["Instagram Reels", "YouTube Shorts", "TikTok", "Korean parenting communities"],
            "native_context": "Korean parent education short-form.",
            "language_note": (
                "Use native Korean phrasing only when it fits the searched trend. "
                "Do not default every concept to 새벽 육아, search anxiety, or an exhausted mother."
            ),
        },
        "spanish": {
            "platforms": ["TikTok Latino", "Instagram Reels", "YouTube Shorts"],
            "native_context": "Spanish-language parent education short-form.",
            "language_note": "Use warm native Spanish phrasing. Avoid stereotype-led templates.",
        },
    },
    "trend_directives": [
        "Treat current search findings as the primary creative input.",
        "Prefer specific observed hook/caption/editing patterns over generic parenting memes.",
        "Capture what is current, and mark weak or stale findings in avoid.",
        "Do not force preselected meme templates when the search results point elsewhere.",
    ],
    "creative_directives": [
        "Generate candidates from the trend research first, then adapt them to the medical script.",
        "Each candidate must cite or clearly echo one searched trend pattern in trend_rationale.",
        "If trend research is empty, use a neutral educational short-form format instead of a culture stereotype.",
        "Medical safety beats humor, virality, and trend fit.",
    ],
    "visual_policy": [
        "Use one video-level caregiver identity across all languages in the same run, but do not reuse the same pose, prop, room, or composition in every scene.",
        "The same visual_style_anchor and character_sheet should be reused for English, Korean, and Spanish unless the user provides a campaign profile that says otherwise.",
        "Character identity should be neutral, well-rested, and brand-safe by default. Character identity should not include an always-present prop.",
        "Each scene should use a distinct concrete action, ordinary prop, shot type, and background. Do not turn medical concepts into infographics or symbols.",
        "Do not add dark circles, exhausted facial features, messy-night imagery, or 3am bedroom scenes unless the medical script specifically requires a night scenario.",
        "Use current trend research for composition, visual metaphor, and editing energy, not for changing the character identity per language.",
    ],
    "image_guardrails": [
        "No visible text, letters, logos, or signs in generated images.",
        "Allow common home-care and pediatric-office props when relevant, such as blank thermometers, plain medicine bottles, bandages, masks, waiting-room chairs, clinic folders, and stethoscopes in the background. Do not show readable labels, dosage numbers, blood, needles entering skin, graphic symptoms, or active procedures.",
        "9:16 vertical illustration suitable for short-form video.",
    ],
    "banned_templates": [
        "Mamá latina",
        "default exhausted mom",
        "dark circles as a character trait",
        "every Korean hook as 새벽 육아 or search anxiety",
        "fear bait",
        "diagnosis bait",
    ],
    "bgm_policy": [
        "BGM must be instrumental only, with no vocals, humming, lyrics, or artist names.",
        "Sound like a trending TikTok/Reels track: ukulele, claps, bouncy indie-pop or Latin-pop groove, immediate catchy hook in the first 2 seconds.",
        "BGM must always be bright, cheerful, and high-energy — never somber, dark, or low-energy regardless of topic.",
        "Target bpm 116-122, brightness 0.90-0.96, density 0.62-0.68 for maximum viral energy.",
    ],
}


def load_campaign_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Load a campaign profile, deep-merging it over the default profile."""
    profile = copy.deepcopy(DEFAULT_CAMPAIGN_PROFILE)
    if not path:
        return profile

    with Path(path).open("r", encoding="utf-8") as handle:
        override = json.load(handle)
    if not isinstance(override, dict):
        raise ValueError("Campaign profile JSON must be an object.")
    return _deep_merge(profile, override)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def profile_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, indent=2)
