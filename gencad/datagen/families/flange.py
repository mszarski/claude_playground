"""Reference family: pipe flanges and hubbed discs (stacked cylinders, bore, bolt-circle holes with optional
counterbore/countersink, fillet at the hub root, chamfered rim).

Conventions every family follows (see datagen/README.md):
  * build in the XY frame so face selectors ('>Z', '<Z') mean what they say; orient the finished part with at most
    one final .rotate((0,0,0),(ax,ay,az),angle) about a coordinate axis;
  * pick edges for fillets/chamfers with near((x,y,z), ...) (mm/cad_helpers.py): points on the edges;
  * literal code only, numbers via f(); part centred at the origin with its longest side ~100-200.
"""
from datagen.families.common import f, on_circle, orient


def generate(rng):
    D = rng.uniform(120, 200)                      # flange outer diameter
    t = rng.uniform(0.08, 0.2) * D                 # flange thickness
    hub_d = rng.uniform(0.35, 0.6) * D
    hub_h = rng.uniform(0.1, 0.45) * D
    bore = rng.uniform(0.35, 0.75) * hub_d
    n_holes = rng.choice([3, 4, 4, 6, 6, 8, 8, 12])
    bc = (D / 2 + hub_d / 2) / 2 * rng.uniform(0.95, 1.05)    # bolt circle between hub and rim
    hole_d = min(rng.uniform(0.06, 0.12) * D, 0.8 * (D / 2 - hub_d / 2))
    z0 = -(t + hub_h) / 2                                       # centre along the axis
    s = f"cq.Workplane('XY').workplane(offset={f(z0)}).circle({f(D / 2)}).extrude({f(t)})"
    s += f".faces('>Z').workplane().circle({f(hub_d / 2)}).extrude({f(hub_h)})"
    if rng.random() < 0.5:                                      # fillet where the hub meets the flange
        s += (f".edges(near({on_circle(hub_d / 2, z0 + t)}))"
              f".fillet({f(min(t, hub_h, (D - hub_d) / 2) * 0.3)})")
    s += f".faces('>Z').workplane().hole({f(bore)})"
    pts = f".faces('<Z').workplane(centerOption='CenterOfBoundBox').polarArray({f(bc)},0,360,{n_holes})"
    k = rng.random()
    if k < 0.6:
        s += pts + f".hole({f(hole_d)})"
    elif k < 0.8:
        s += pts + f".cboreHole({f(hole_d)},{f(hole_d * 1.7)},{f(t * 0.3)})"
    else:
        s += pts + f".cskHole({f(hole_d)},{f(hole_d * 1.8)},90)"
    if rng.random() < 0.6:                                      # chamfer the outer rim on the bottom face
        s += f".edges(near({on_circle(D / 2, z0)})).chamfer({f(t * 0.15)})"
    return "import cadquery as cq\nr=" + s + orient(rng)
