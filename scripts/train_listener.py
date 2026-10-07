"""Train the learned listener (rmr.listen_model) on the Seamless Interaction features (scripts/extract_listening.py).

  python scripts/train_listener.py --data runs/listening/seamless/v1 --out runs/listener

Each conversation gives two examples (either person as the listener). Loss only on frames where the listener is
silent and tracked; the speaker features are computed causally from the partner's own channel. Writes
``listener.pt``, ``listener.json`` (weights for rmr.listen_model.LearnedHead and visualizer/src/ListenModel.js)
and ``log.json``.
"""
import argparse
import json
import os
import time

import numpy as np


def load_examples(data, split):
    from rmr.listen_model import fast_motion, speaker_features

    pairs = [p for p in json.load(open(os.path.join(data, "pairs.json"))) if p[0] == split]
    out = []
    for _, a, b in pairs:
        try:
            za, zb = (np.load(os.path.join(data, split, f + ".npz")) for f in (a, b))
        except FileNotFoundError:
            continue
        for spk, lis in ((za, zb), (zb, za)):
            T = min(len(spk["db"]), len(lis["rot"]))
            valid = lis["valid"][:T]
            y = fast_motion(lis["rot"][:T], valid)
            # trust the target only well inside tracked stretches (the 0.3 Hz filter bleeds across gaps)
            ok = valid.copy()
            for s in range(1, 13):
                ok[s:] &= valid[:-s]
                ok[:-s] &= valid[s:]
            mask = ok & ~lis["vad"][:T]
            if mask.sum() < 250:
                continue
            out.append({"x": speaker_features(spk["db"][:T], spk["vad"][:T]), "y": y, "mask": mask,
                        "spk_vad": spk["vad"][:T].astype(bool), "db": spk["db"][:T]})
    return out


def batches(examples, win, bs, rng):
    import torch

    from rmr.listen_model import DMAX, to_bins

    starts = [(i, s) for i, e in enumerate(examples) for s in range(0, len(e["x"]) - win, win // 2)
              if e["mask"][s:s + win].mean() > 0.3]
    rng.shuffle(starts)
    for k in range(0, len(starts) - bs + 1, bs):
        X, Y, M = [], [], []
        for i, s in starts[k:k + bs]:
            e = examples[i]
            y = e["y"][s - 1 if s else 0:s + win]
            y = np.concatenate([y[:1], y]) if s == 0 else y
            d = np.diff(y, axis=0)                       # d[t] = y[t] - y[t-1], t = s..s+win-1
            prev_y, prev_d = y[:-1], np.concatenate([d[:1] * 0, d[:-1]])
            X.append(np.concatenate([e["x"][s:s + win], prev_y / 10, prev_d / DMAX], -1))
            Y.append(to_bins(d))
            M.append(e["mask"][s:s + win])
        yield (torch.tensor(np.stack(X)), torch.tensor(np.stack(Y)), torch.tensor(np.stack(M)))


def nll(net, examples, win, bs):
    import torch

    tot, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for X, Y, M in batches(examples, win, bs, np.random.default_rng(0)):
            lg, _ = net(X)
            ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y.reshape(-1), reduction="none")
            ce = ce.view(*Y.shape)[M]
            tot, n = tot + ce.sum().item(), n + ce.numel()
    net.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/listening/seamless/v1")
    ap.add_argument("--out", default="runs/listener")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--win", type=int, default=375)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    import torch

    from rmr.listen_model import export, make_net

    torch.manual_seed(0)
    train, dev = load_examples(a.data, "train"), load_examples(a.data, "dev")
    def hours(ex):
        return sum(e["mask"].sum() for e in ex) / 25 / 3600

    print(f"train {len(train)} listeners, {hours(train):.1f} h listening | dev {len(dev)}, {hours(dev):.1f} h", flush=True)
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    log, best = [], 1e9
    os.makedirs(a.out, exist_ok=True)
    print(f"dev nll before training {nll(net, dev, a.win, a.bs):.3f} (uniform {np.log(31):.3f})", flush=True)
    for ep in range(a.epochs):
        t0, tl, nb = time.time(), 0.0, 0
        for g in opt.param_groups:
            g["lr"] = a.lr * 0.5 * (1 + np.cos(np.pi * ep / a.epochs))
        for X, Y, M in batches(train, a.win, a.bs, rng):
            lg, _ = net(X)
            ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y.reshape(-1), reduction="none")
            loss = ce.view(*Y.shape)[M].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl, nb = tl + loss.item(), nb + 1
        d = nll(net, dev, a.win, a.bs)
        log.append({"epoch": ep + 1, "train": tl / nb, "dev": d})
        print(f"epoch {ep + 1}: train {tl / nb:.3f} dev {d:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if d < best:
            best = d
            torch.save(net.state_dict(), os.path.join(a.out, "listener.pt"))
            with open(os.path.join(a.out, "listener.json"), "w") as f:
                json.dump(export(net), f)
    with open(os.path.join(a.out, "log.json"), "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
