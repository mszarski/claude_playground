"""Mass generation of rich-feature parts (datagen/families): generate, check (datagen.check), render with our 4
fixed cameras (scripts/render_views.py), write one shard.

  python scripts/gen_rich_data.py --shard 0 --n_shards 32 --per_shard 30000 --out data/rich/train
  python scripts/gen_rich_data.py --split test --shard 0 --n_shards 1 --per_shard 1000 --out data/rich/test

Seeds are disjoint by construction: seed = hash(split, family, shard, k), so train/test never share a program (and
programs are also deduplicated by text inside a shard). --families picks a subset with weights (name=weight,...);
--exclude holds families out (for a held-out-family test). Output: <out>/<shard>/renders/<uid>_{0..3}.png and
<out>/<shard>/parts.parquet (uid, family, code, faces, volume). A CPU job tars the shard dir into the bucket.
"""
import argparse
import hashlib
import importlib
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT = ("flange=1,l_bracket=1,u_bracket=1,gusset_mount=1,mount_plate=1,pocket_plate=1,bar_link=1,stepped_shaft=1,"
           "pulley=1,bushing=1,enclosure=1,bearing_block=1,motor_mount=1,hex_hardware=1,knob_handle=1,gear_like=0.5,"
           "extrusion_profile=1,bent_sheet=1,rib_panel=1,random_mix=3,random_mix_hard=2")


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def make(args):
    """Worker: generate one part for (split, family, shard, k); check; render. -> record or None."""
    split, fam, shard, k, out = args
    from datagen.check import build, check
    from scripts.render_views import DIRS, render, tessellate_cq
    mod = importlib.import_module(f"datagen.families.{fam}")
    code = mod.generate(random.Random(seed_of(split, fam, shard, k)))
    ok, info = check(code)
    if not ok:
        return {"ok": False, "family": fam, "reason": info["reason"][:60]}
    uid = f"{fam}/{split}_{shard}_{k}"
    stem = os.path.join(out, "renders", uid)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    tris, fid = tessellate_cq(build(code))
    for j, d in enumerate(DIRS):
        render(tris, fid, d).save(f"{stem}_{j}.png", optimize=True)
    return {"ok": True, "uid": uid, "family": fam, "code": code, "faces": info["faces"], "volume": info["volume"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--per_shard", type=int, default=1000, help="attempted parts per shard (failures are dropped)")
    ap.add_argument("--families", default=DEFAULT)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    from pebble import ProcessPool
    fams = {k: float(v) for k, v in (kv.split("=") for kv in a.families.split(","))}
    for x in filter(None, a.exclude.split(",")):
        fams.pop(x, None)
    names = list(fams)
    w = [fams[n] for n in names]
    rng = random.Random(seed_of("plan", a.split, a.shard))
    plan = [(a.split, rng.choices(names, w)[0], a.shard, k) for k in range(a.per_shard)]
    out = os.path.join(a.out, str(a.shard))
    os.makedirs(out, exist_ok=True)
    recs, fails, seen, t0 = [], {}, set(), time.time()
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [pool.schedule(make, args=((*p, out),), timeout=a.timeout) for p in plan]
        for i, fu in enumerate(futs):
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                r = {"ok": False, "family": plan[i][1], "reason": f"timeout/crash {type(e).__name__}"}
            if r["ok"] and r["code"] not in seen:
                seen.add(r["code"])
                recs.append(r)
            elif not r["ok"]:
                key = f"{r['family']}: {r['reason']}"
                fails[key] = fails.get(key, 0) + 1
            if i % 2000 == 0:
                print(f"[gen] {i}/{len(plan)} kept {len(recs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ok"} for r in recs])
    df.to_parquet(os.path.join(out, "parts.parquet"))
    print(f"[gen] shard {a.shard}: kept {len(df)} / {len(plan)} in {time.time() - t0:.0f}s; per family "
          f"{df.family.value_counts().to_dict() if len(df) else {}}; top failures "
          f"{dict(sorted(fails.items(), key=lambda kv: -kv[1])[:6])}", flush=True)


if __name__ == "__main__":
    main()
