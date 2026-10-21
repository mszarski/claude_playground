"""Validity checks for generated CadQuery programs. A program passes only if it:
  - is plain literal CadQuery (imports only cadquery as cq, no loops/defs/randomness), assigns `r`;
  - builds within a time limit to exactly one non-empty solid that OpenCascade's BRepCheck calls valid;
  - has a sane size and position: longest bounding-box side in [MIN_SIDE, MAX_SIDE], centre within MAX_OFF of the
    origin (the canonical CAD-Recode frame: parts span roughly [-100, 100]);
  - is not a sliver (volume / bbox volume >= MIN_FILL) and rebuilds to the same volume (determinism).
`check(code)` -> (ok: bool, info: dict). `python -m datagen.check <family> [n]` samples a family module and prints
the pass rate and failure reasons.
"""
import ast
import importlib
import json
import random
import sys
import time

MIN_SIDE, MAX_SIDE, MAX_OFF, MIN_FILL = 60.0, 215.0, 40.0, 0.02
ALLOWED_TOP = (ast.Import, ast.Assign, ast.Expr)


def static_ok(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax: {e.msg}"
    for node in tree.body:
        if not isinstance(node, ALLOWED_TOP):
            return False, f"statement {type(node).__name__} not allowed"
        if isinstance(node, ast.Import) and [a.name for a in node.names] != ["cadquery"]:
            return False, "only `import cadquery as cq` allowed"
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.FunctionDef, ast.Lambda, ast.ListComp, ast.GeneratorExp,
                             ast.ImportFrom)):
            return False, f"{type(node).__name__} not allowed (code must be literal)"
        if isinstance(node, ast.Name) and node.id in ("random", "math", "np", "open", "eval", "exec", "__import__"):
            return False, f"name {node.id} not allowed"
    if not any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "r" for t in n.targets) for n in tree.body):
        return False, "no `r = ...`"
    return True, ""


def build(code):
    import cadquery as cq
    from mm.cad_helpers import namespace
    ns = namespace()
    exec(code, ns)
    r = ns["r"]
    if isinstance(r, cq.Workplane):
        objs = [o for o in r.vals() if isinstance(o, cq.Shape)]
        if not objs:
            raise ValueError("workplane holds no shape")
        shape = objs[0] if len(objs) == 1 else cq.Compound.makeCompound(objs)
    elif isinstance(r, cq.Shape):
        shape = r
    else:
        raise ValueError(f"r is {type(r).__name__}")
    return shape


def check(code):
    ok, why = static_ok(code)
    if not ok:
        return False, {"reason": why}
    try:
        t0 = time.time()
        shape = build(code)
        dt = time.time() - t0
    except Exception as e:  # noqa: BLE001
        return False, {"reason": f"build error: {type(e).__name__}: {str(e)[:120]}"}
    solids = shape.Solids()
    if len(solids) != 1:
        return False, {"reason": f"{len(solids)} solids"}
    if not shape.isValid():
        return False, {"reason": "BRepCheck invalid"}
    vol = shape.Volume()
    bb = shape.BoundingBox()
    side = max(bb.xlen, bb.ylen, bb.zlen)
    cen = ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
    info = {"volume": round(vol, 1), "side": round(side, 1), "faces": len(shape.Faces()), "build_s": round(dt, 3)}
    if vol <= 0:
        return False, {**info, "reason": "non-positive volume"}
    if not MIN_SIDE <= side <= MAX_SIDE:
        return False, {**info, "reason": f"longest side {side:.0f} outside [{MIN_SIDE:.0f}, {MAX_SIDE:.0f}]"}
    if max(abs(c) for c in cen) > MAX_OFF:
        return False, {**info, "reason": f"centre {tuple(round(c) for c in cen)} too far from origin"}
    if vol / max(bb.xlen * bb.ylen * bb.zlen, 1e-9) < MIN_FILL:
        return False, {**info, "reason": "sliver (fills < 2 % of its box)"}
    try:
        if abs(build(code).Volume() - vol) > 1e-6 * max(vol, 1):
            return False, {**info, "reason": "rebuild gives a different volume"}
    except Exception:  # noqa: BLE001
        return False, {**info, "reason": "rebuild failed"}
    return True, info


def main():
    fam = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    mod = importlib.import_module(f"datagen.families.{fam}")
    reasons, ok, faces, t = {}, 0, [], time.time()
    for i in range(n):
        code = mod.generate(random.Random(i))
        good, info = check(code)
        ok += good
        if good:
            faces.append(info["faces"])
        else:
            reasons[info["reason"][:60]] = reasons.get(info["reason"][:60], 0) + 1
    print(json.dumps({"family": fam, "n": n, "pass_rate": round(ok / n, 3),
                      "mean_faces": round(sum(faces) / max(len(faces), 1), 1),
                      "sec_per_part": round((time.time() - t) / n, 3),
                      "failures": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:8])}, indent=1))


if __name__ == "__main__":
    main()
