"""Compare listening head motion on held-out conversations: real listeners, rmr.listen's rules, the learned model.

  python scripts/eval_listener.py --data runs/listening/seamless/v1 --weights runs/listener/listener.json

On the dev split's listening frames (listener silent and tracked), for the fast part of head motion
(rmr.listen_model.fast_motion; the rules' output goes through the same filter):
* how much the head moves: RMS angle per axis (deg) and mean pitch speed (deg/s);
* nods per minute: peaks of downward pitch at least 1.5 deg prominent;
* timing: share of nods that start within 1 s after the speaker pauses (voice activity ends after >= 1 s of speech),
  and the same share for random times ("chance"), so 2x chance means nods follow pauses twice as often as luck.
"""
import argparse
import json

import numpy as np


def nods(pitch, mask):
    from scipy.signal import find_peaks

    pk, _ = find_peaks(pitch, prominence=1.5, distance=8)
    return pk[mask[pk]]


def after_pause(spk_vad, win=25):
    """Frames within ``win`` frames after the speaker stops, having spoken >= 1 s."""
    out = np.zeros(len(spk_vad), bool)
    run = 0
    for t in range(1, len(spk_vad)):
        run = run + 1 if spk_vad[t - 1] else 0
        if spk_vad[t - 1] and not spk_vad[t] and run >= 25:
            out[t:t + win] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/listening/seamless/v1")
    ap.add_argument("--weights", default="runs/listener/listener.json")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, "scripts")
    from train_listener import load_examples

    from rmr.listen import Listener
    from rmr.listen_model import LearnedHead, fast_motion

    w = json.load(open(a.weights))
    stats = {k: {"sq": np.zeros(3), "speed": 0.0, "n": 0, "nods": 0, "nods_after": 0, "minutes": 0.0,
                 "sp_after": 0.0, "n_after": 0, "sp_talk": 0.0, "n_talk": 0}
             for k in ("real", "rules", "learned")}
    chance_num = chance_den = 0
    for i, e in enumerate(load_examples(a.data, "dev")):
        m, sv = e["mask"], e["spk_vad"]
        motions = {"real": e["y"]}
        # rules: driven by the speaker's loudness (the raw channel, as the robot's mic would hear it)
        lis = Listener()
        A = np.array([lis.step(float(v)) for v in e["db"]])
        rules = np.degrees(A[:, [4, 5, 3]])     # pitch, yaw, roll
        motions["rules"] = fast_motion(np.radians(rules), np.ones(len(rules), bool))
        head = LearnedHead(w, temperature=a.temperature, seed=i)
        motions["learned"] = np.array([head.step(f) for f in e["x"]])
        ap_mask = after_pause(sv)
        chance_num += (ap_mask & m).sum()
        chance_den += m.sum()
        for k, y in motions.items():
            s = stats[k]
            s["sq"] += (y[m] ** 2).sum(0)
            v = np.abs(np.diff(y[:, 0])) * 25
            s["speed"] += v[m[1:]].sum()
            aft, talk = (ap_mask & m)[1:], (sv & m)[1:]
            s["sp_after"] += v[aft].sum()
            s["n_after"] += aft.sum()
            s["sp_talk"] += v[talk].sum()
            s["n_talk"] += talk.sum()
            s["n"] += m.sum()
            pk = nods(y[:, 0], m)
            s["nods"] += len(pk)
            s["nods_after"] += ap_mask[pk].sum()
            s["minutes"] += m.sum() / 25 / 60
    chance = chance_num / chance_den
    rows = {}
    print(f"{'':8s} {'RMS pitch/yaw/roll deg':>24s} {'pitch speed':>12s} {'nods/min':>9s} {'after pause':>12s}"
          f" {'moves after pause vs during speech':>36s}")
    for k, s in stats.items():
        rms = np.sqrt(s["sq"] / s["n"])
        after = s["nods_after"] / max(s["nods"], 1)
        resp = (s["sp_after"] / max(s["n_after"], 1)) / max(s["sp_talk"] / max(s["n_talk"], 1), 1e-9)
        rows[k] = {"rms_deg": rms.round(2).tolist(), "pitch_speed_deg_s": round(s["speed"] / s["n"], 1),
                   "nods_per_min": round(s["nods"] / s["minutes"], 1), "nods_after_pause": round(after, 3),
                   "speed_after_pause_vs_speech": round(resp, 2)}
        print(f"{k:8s} {' / '.join(f'{v:4.2f}' for v in rms):>24s} {s['speed'] / s['n']:9.1f} °/s "
              f"{s['nods'] / s['minutes']:9.1f} {after:7.0%} ({after / chance:.1f}x chance) {resp:20.2f}x")
    print(f"chance: {chance:.0%} of listening time is within 1 s after a pause; "
          f"{stats['real']['minutes']:.0f} min of listening")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"chance": chance, **rows}, f, indent=1)


if __name__ == "__main__":
    main()
