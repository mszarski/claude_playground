"""Shared helpers for part families."""


def f(x):
    """Format a number compactly: an integer when it is one after rounding to 0.1, else one decimal."""
    x = round(float(x), 1)
    return str(int(x)) if x == int(x) else str(x)


def orient(rng, p_keep=0.4):
    """Optional final rotation of an XY-built part about a coordinate axis (by 90/180/270 degrees), as code."""
    if rng.random() < p_keep:
        return ""
    axis = rng.choice(["(1,0,0)", "(0,1,0)", "(0,0,1)"])
    ang = rng.choice([90, 90, 180, 270])
    return f".rotate((0,0,0),{axis},{ang})"


def on_circle(r, z, cx=0.0, cy=0.0, deg=37.0):
    """A point on a horizontal circle (centre cx, cy, height z) for near(...), away from the seam OpenCascade puts at
    angle 0 on every cylinder (a point on the seam is equally close to the seam line and the circle)."""
    import math
    a = math.radians(deg)
    return f"({f(cx + r * math.cos(a))},{f(cy + r * math.sin(a))},{f(z)})"
