"""Design-system context for prompts: the file's own conventions (colors, component states, type and spacing scales).

In a real file the model can read the design system (variables, styles, components), so a request like "disable the
Save button" has a defined answer. Our synthetic test sets never gave it, which turned some edit types into
guessing the generator's taste. Eval v2 (and training data built for it) puts one of these blocks in the prompt.
"""

from __future__ import annotations

import random
from typing import Any

BRANDS = [("Blue", "#2563EB", "#1D4ED8"), ("Indigo", "#4F46E5", "#4338CA"), ("Violet", "#7C3AED", "#6D28D9"),
          ("Emerald", "#059669", "#047857"), ("Teal", "#0D9488", "#0F766E"), ("Orange", "#EA580C", "#C2410C"),
          ("Rose", "#E11D48", "#BE123C"), ("Sky", "#0284C7", "#0369A1")]
NEUTRALS = [  # text primary, secondary, disabled, border, surface, page
    ("#111827", "#6B7280", "#9CA3AF", "#E5E7EB", "#FFFFFF", "#F9FAFB"),
    ("#0F172A", "#64748B", "#94A3B8", "#E2E8F0", "#FFFFFF", "#F8FAFC"),
    ("#18181B", "#71717A", "#A1A1AA", "#E4E4E7", "#FFFFFF", "#FAFAFA"),
]
DANGER = ["#DC2626", "#E11D48"]


def sample_design_system(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    name, brand, brand_dark = rng.choice(BRANDS)
    text, text2, text_dis, border, surface, page = rng.choice(NEUTRALS)
    radius = rng.choice([4, 6, 8, 12])
    disabled = rng.choice(["opacity", "grey"])
    return {
        "brand": brand, "brand_dark": brand_dark, "brand_name": name, "danger": rng.choice(DANGER),
        "text": text, "text_secondary": text2, "text_disabled": text_dis, "border": border, "surface": surface,
        "page": page, "radius": radius, "disabled": disabled,
        "type": rng.choice([(28, 20, 16, 14, 12), (32, 24, 16, 14, 12), (24, 18, 15, 13, 12)]),
        "spacing": rng.choice([(4, 8, 12, 16, 24, 32), (4, 8, 16, 24, 32, 48)]),
        "selected": rng.choice(["stroke", "fill"]),
    }


def render_design_system(ds: dict[str, Any]) -> str:
    h1, h2, body, small, caption = ds["type"]
    dis = ("opacity=0.5, otherwise unchanged" if ds["disabled"] == "opacity"
           else f"fill={ds['border']} no stroke, label color={ds['text_disabled']}")
    sel = (f"stroke={ds['brand']}" if ds["selected"] == "stroke" else f"fill={ds['brand']}, label color=#FFFFFF")
    return "\n".join([
        f"Colors: brand {ds['brand']} (hover/pressed {ds['brand_dark']}), danger {ds['danger']}, "
        f"text {ds['text']}, secondary text {ds['text_secondary']}, disabled text {ds['text_disabled']}, "
        f"border {ds['border']}, surface {ds['surface']}, page {ds['page']}.",
        f"Type scale (size/weight): heading {h1}/700, subheading {h2}/600, body {body}/400, "
        f"label {small}/500, caption {caption}/400.",
        f"Spacing scale: {', '.join(map(str, ds['spacing']))}. Corner radius: {ds['radius']} (pills 999).",
        f"Buttons: primary = fill={ds['brand']} no stroke, label color=#FFFFFF weight=600; "
        f"secondary = fill={ds['surface']} stroke={ds['border']}, label color={ds['text']}; "
        f"destructive = fill={ds['danger']} no stroke, label color=#FFFFFF; disabled = {dis}.",
        f"Selected/active item (tab, option, plan, toggle on): {sel}. Inputs: fill={ds['surface']} "
        f"stroke={ds['border']}; invalid input stroke={ds['danger']} with a caption-size error text in "
        f"{ds['danger']} below it.",
    ])


def user_message_ds(design_text: str, ds_text: str, instruction: str) -> str:
    return f"Design system:\n{ds_text}\n\nDesign:\n{design_text}\n\nRequest: {instruction}"
