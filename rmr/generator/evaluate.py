"""Generator check on held-out real clips, generated from their true (extracted) plans.

- identification: is each generated motion closest to its own real clip among the held-out set?
  (chance = 1 / number of held-out clips)
- speed: 95th-percentile and peak head-pitch / antenna speeds, generated vs real
  (too slow = sluggish, too fast = jittery)
- energy_hold: generated / requested fast-detail energy on a held pose with constant energy, as recipes write it
  (``hold 5 E=6``); 1 = the energy channel is followed. Real plans hide the problem: their posture curves carry
  traces of the detail, which the reference's training lets the generator read instead of the energy channel.

Reference: ``generator/evaluate.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import numpy as np

from .. import plan as PL
from ..motion import FPS, move_to_traj

NT = 64


class HeldOut:
    """``rank(A, name)``: where clip ``name`` ranks among the held-out clips by distance to trajectory ``A``
    (time-shift-tolerant RMS over resampled, z-scored 9-DoF trajectories); 1 = identified."""

    def __init__(self, moves, held_out):
        trajs = {n: move_to_traj(m) for n, m in moves}
        allA = np.concatenate(list(trajs.values()))
        self.MU, self.SD = allA.mean(0), allA.std(0) + 1e-6
        self.real = {h: trajs[h] for h in held_out if h in trajs}
        self.feats = {h: self.feat(A) for h, A in self.real.items()}

    def feat(self, A):
        u, v = np.linspace(0, 1, NT), np.linspace(0, 1, len(A))
        return (np.stack([np.interp(u, v, A[:, j]) for j in range(9)], -1) - self.MU) / self.SD

    @staticmethod
    def dist(a, b, shift=4):
        return min(float(np.sqrt(((a[max(0, s):NT + min(0, s)] - b[max(0, -s):NT - max(0, s)]) ** 2).mean()))
                   for s in range(-shift, shift + 1))

    def rank(self, A, name):
        f = self.feat(A)
        d = {k: self.dist(f, v) for k, v in self.feats.items()}
        return sorted(d, key=d.get).index(name) + 1


def speeds(A):
    pitch = np.abs(np.diff(np.degrees(A[:, 4]))) * FPS
    ear = np.abs(np.diff(np.degrees(A[:, 6:8]), axis=0)).max(1) * FPS
    return np.percentile(pitch, 95), pitch.max(), np.percentile(ear, 95), ear.max()


def energy_hold(net, stats, dev, levels=(2, 4, 6, 8), seeds=3, steps=8):
    """``{level: generated / requested energy}`` over the hold, with the sampler's default settings."""
    from ..recipe import variants
    from .sample import generate_batch

    out = {}
    for e in levels:
        plans = [variants(f"go .5 e=40 p=8 z=-4 E={e} | hold 5 E={e}", 1, seed=s)[0] for s in range(seeds)]
        G = generate_batch(net, stats, plans, dev, seeds=list(range(seeds)), steps=steps)
        got = [PL.frames(PL.extract(A), len(A))[FPS:, 7].mean() for A in G]
        req = [PL.frames(p, len(A))[FPS:, 7].mean() for p, A in zip(plans, G)]
        out[str(e)] = float(np.mean(got) / np.mean(req))
    return out


def evaluate(net, stats, dev, moves, held_out, seeds=3, steps=100):
    from .sample import generate

    H = HeldOut(moves, held_out)
    ranks, gen_speeds, real_speeds = [], [], []
    for name, A in H.real.items():
        plan = PL.extract(A)
        real_speeds.append(speeds(A))
        for s in range(seeds):
            G = generate(net, stats, plan, dev, seed=s, steps=steps)
            gen_speeds.append(speeds(G))
            ranks.append(H.rank(G, name))
    r = np.array(ranks)
    g, q = np.mean(gen_speeds, 0), np.mean(real_speeds, 0)
    return {"n_clips": len(H.real), "n": len(r), "top1": float((r == 1).mean()), "mean_rank": float(r.mean()),
            "chance": 1 / max(1, len(H.real)),
            "pitch_speed_p95": [float(g[0]), float(q[0])], "pitch_speed_peak": [float(g[1]), float(q[1])],
            "ear_speed_p95": [float(g[2]), float(q[2])], "ear_speed_peak": [float(g[3]), float(q[3])],
            "energy_hold": energy_hold(net, stats, dev)}
