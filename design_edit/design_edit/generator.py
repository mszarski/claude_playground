"""Procedural generator for small, realistic UI designs.

Designs are built from component templates (cards, forms, navbars, pricing,
settings lists), so every node has a semantic ``name`` that instructions can
refer to. Values come from fixed spacing, type and color scales. This is what
makes "the off-scale value" or "the low-contrast text" well defined in
:mod:`design_edit.edits`.
"""

from __future__ import annotations

import random
from typing import Any

from .schema import Node, make_node

SPACING = [4, 8, 12, 16, 20, 24, 32, 40, 48]
TYPE_SCALE = [12, 14, 16, 18, 20, 24, 30, 36, 48]
RADII = [0, 4, 6, 8, 12, 16, 24]
WEIGHTS = [400, 500, 600, 700]

BRANDS = {
    "blue": "#2563EB", "indigo": "#4F46E5", "violet": "#7C3AED", "green": "#16A34A",
    "teal": "#0D9488", "orange": "#EA580C", "red": "#DC2626", "pink": "#DB2777",
    "black": "#111827",
}
NEUTRAL = {"white": "#FFFFFF", "bg": "#F9FAFB", "border": "#E5E7EB", "muted": "#6B7280",
           "text": "#111827", "subtle": "#F3F4F6"}

TITLES = ["Welcome back", "Your weekly summary", "Upgrade your plan", "Invite your team",
          "Project settings", "New message", "Start your free trial", "Order confirmed",
          "Choose a template", "Billing overview", "Create an account", "Recent activity"]
BODIES = ["Here's what happened while you were away.", "Get unlimited projects and priority support.",
          "Collaborators can edit files and leave comments.", "Your changes will be saved automatically.",
          "We'll send a receipt to your inbox.", "Pick a starting point or begin from scratch.",
          "Track usage and manage payment methods.", "Join thousands of teams shipping faster."]
PRIMARY_LABELS = ["Continue", "Get started", "Save changes", "Upgrade", "Send invite", "Sign in",
                  "Create project", "Confirm", "Subscribe", "Next"]
SECONDARY_LABELS = ["Cancel", "Learn more", "Back", "Skip", "Not now", "Preview"]
NAV_LINKS = ["Product", "Pricing", "Docs", "Blog", "Customers", "Company", "Changelog"]
PLANS = [("Free", "$0"), ("Starter", "$9"), ("Pro", "$24"), ("Team", "$49"), ("Enterprise", "Custom")]
FEATURES = ["Unlimited projects", "Version history", "Custom domains", "Priority support",
            "SSO & SAML", "Advanced analytics", "Shared libraries", "Offline mode"]
SETTINGS = ["Email notifications", "Dark mode", "Two-factor auth", "Weekly digest",
            "Auto-save", "Show grid", "Snap to pixel"]
FIELDS = ["Email", "Password", "Full name", "Company", "Workspace URL"]


class IdGen:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"n{self.n}"


class Ctx:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.id = IdGen()
        self.brand_name, self.brand = rng.choice(list(BRANDS.items()))
        self.radius = rng.choice([4, 6, 8, 12])
        self.pad = rng.choice([16, 20, 24, 32])
        self.gap = rng.choice([8, 12, 16])

    def text(self, name: str, text: str, size: int, weight: int = 400, color: str | None = None) -> Node:
        return make_node(self.id(), "text", name=name, text=text, size=size, weight=weight,
                         color=color or NEUTRAL["text"])

    def button(self, label: str, primary: bool = True, name: str | None = None) -> Node:
        nid = self.id()
        txt = self.text(f"{label} label", label, 14, 600, NEUTRAL["white"] if primary else NEUTRAL["text"])
        props: dict[str, Any] = dict(name=name or f"{label} button", layout="row", pad=12, align="center",
                                     justify="center", radius=self.radius)
        if primary:
            props["fill"] = self.brand
        else:
            props.update(fill=NEUTRAL["white"], stroke="#D1D5DB")
        return make_node(nid, "frame", [txt], **props)


# ---------------------------------------------------------------------------
# Component templates
# ---------------------------------------------------------------------------

def card(c: Ctx) -> Node:
    nid = c.id()
    kids: list[Node] = []
    if c.rng.random() < 0.5:
        kids.append(make_node(c.id(), "image", name="Hero image", h=160, radius=max(c.radius - 4, 0),
                              fill=NEUTRAL["subtle"]))
    kids.append(c.text("Title", c.rng.choice(TITLES), c.rng.choice([20, 24]), 600))
    kids.append(c.text("Body", c.rng.choice(BODIES), 14, 400, NEUTRAL["muted"]))
    actions = [c.button(c.rng.choice(PRIMARY_LABELS))]
    if c.rng.random() < 0.6:
        actions.append(c.button(c.rng.choice(SECONDARY_LABELS), primary=False))
    kids.append(make_node(c.id(), "frame", actions, name="Actions", layout="row", gap=8))
    return make_node(nid, "frame", kids, name="Card", layout="col", w=360, gap=c.gap, pad=c.pad,
                     fill=NEUTRAL["white"], stroke=NEUTRAL["border"], radius=c.radius + 4)


def form(c: Ctx) -> Node:
    nid = c.id()
    kids = [c.text("Heading", c.rng.choice(["Sign in", "Create an account", "Join the beta"]), 24, 700),
            c.text("Subheading", c.rng.choice(BODIES), 14, 400, NEUTRAL["muted"])]
    for f in c.rng.sample(FIELDS, c.rng.randint(1, 3)):
        fid = c.id()
        label = c.text(f"{f} label", f, 14, 500)
        inp = make_node(c.id(), "frame", [c.text(f"{f} placeholder", f"Enter {f.lower()}", 14, 400, "#9CA3AF")],
                        name=f"{f} input", layout="row", pad=12, radius=c.radius, fill=NEUTRAL["white"],
                        stroke="#D1D5DB")
        kids.append(make_node(fid, "frame", [label, inp], name=f"{f} field", layout="col", gap=6))
    kids.append(c.button(c.rng.choice(["Sign in", "Continue", "Create account"])))
    if c.rng.random() < 0.5:
        kids.append(c.text("Footer link", "Forgot your password?", 14, 500, c.brand))
    return make_node(nid, "frame", kids, name="Form", layout="col", w=400, gap=c.gap + 4, pad=c.pad + 8,
                     fill=NEUTRAL["white"], radius=c.radius + 4, stroke=NEUTRAL["border"])


def navbar(c: Ctx) -> Node:
    nid = c.id()
    logo = c.text("Logo", c.rng.choice(["Acme", "Northwind", "Lumen", "Orbit", "Parcel"]), 20, 700)
    links = make_node(c.id(), "frame",
                      [c.text(f"{l} link", l, 14, 500, NEUTRAL["muted"]) for l in c.rng.sample(NAV_LINKS, c.rng.randint(3, 5))],
                      name="Links", layout="row", gap=24, align="center")
    cta = c.button(c.rng.choice(["Get started", "Sign up", "Try free"]))
    return make_node(nid, "frame", [logo, links, cta], name="Navbar", layout="row", w=1200, pad=16,
                     align="center", justify="between", fill=NEUTRAL["white"], stroke=NEUTRAL["border"])


def pricing(c: Ctx) -> Node:
    nid = c.id()
    plans = c.rng.sample(PLANS, c.rng.randint(2, 3))
    plans.sort(key=lambda p: PLANS.index(p))
    cards = []
    for i, (plan, price) in enumerate(plans):
        pid = c.id()
        feats = make_node(c.id(), "frame",
                          [c.text(f"{plan} feature {j + 1}", f, 14, 400, NEUTRAL["muted"])
                           for j, f in enumerate(c.rng.sample(FEATURES, 3))],
                          name=f"{plan} features", layout="col", gap=8)
        highlight = i == len(plans) - 1
        cards.append(make_node(pid, "frame", [
            c.text(f"{plan} name", plan, 18, 600),
            c.text(f"{plan} price", price, 36, 700),
            feats,
            c.button(f"Choose {plan}", primary=highlight, name=f"{plan} button"),
        ], name=f"{plan} plan", layout="col", w=280, gap=c.gap, pad=c.pad, radius=c.radius + 4,
            fill=NEUTRAL["white"], stroke=c.brand if highlight else NEUTRAL["border"]))
    return make_node(nid, "frame", cards, name="Pricing", layout="row", gap=24, align="start")


def settings(c: Ctx) -> Node:
    nid = c.id()
    rows = [c.text("Section title", c.rng.choice(["Preferences", "Notifications", "Security"]), 18, 600)]
    for s in c.rng.sample(SETTINGS, c.rng.randint(2, 4)):
        rid = c.id()
        on = c.rng.random() < 0.5
        rows.append(make_node(rid, "frame", [
            c.text(f"{s} label", s, 14, 500),
            make_node(c.id(), "rect", name=f"{s} toggle", w=36, h=20, radius=10,
                      fill=c.brand if on else "#D1D5DB"),
        ], name=f"{s} row", layout="row", justify="between", align="center", pad=12))
    return make_node(nid, "frame", rows, name="Settings", layout="col", w=480, gap=4, pad=c.pad,
                     fill=NEUTRAL["white"], radius=c.radius + 4, stroke=NEUTRAL["border"])


COMPONENTS = {"card": card, "form": form, "navbar": navbar, "pricing": pricing, "settings": settings}


def generate_design(seed: int, max_sections: int = 2) -> Node:
    """Generate a page with 1..max_sections components. Deterministic in ``seed``."""
    rng = random.Random(seed)
    c = Ctx(rng)
    root_id = c.id()
    kinds = rng.sample(list(COMPONENTS), rng.randint(1, max_sections))
    if "navbar" in kinds:  # navbars go on top
        kinds.remove("navbar")
        kinds.insert(0, "navbar")
    sections = [COMPONENTS[k](c) for k in kinds]
    return make_node(root_id, "frame", sections, name="Page", layout="col", w=1280, gap=32, pad=40,
                     align="center", fill=NEUTRAL["bg"])
