"""Plans -> 25 Hz trajectories with a trained generator.

Reference: ``generator/sample.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from .. import plan as PL
from ..motion import FPS
from .model import MotionGenerator, device


def load(ckpt, dev=None):
    """Checkpoint -> ``(net, stats, device)``."""
    dev = dev or device()
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    net = MotionGenerator(**ck.get("config", {})).to(dev)
    net.load_state_dict(ck["sd"])
    net.eval()
    return net, ck["stats"], dev


@torch.no_grad()
def generate_batch(net, stats, plans, dev, seeds=None, steps=8, cfg=1.5, lowpass_hz=4.0):
    """Plans -> list of ``(T_i, 9)`` trajectories.

    Euler integration of the flow from noise (t=1) to data (t=0) with classifier-free guidance ``cfg`` on
    the plan, then a low-pass at ``lowpass_hz`` (the robot's useful bandwidth). All plans (padded, masked)
    and both guidance branches share one forward pass per step. Duration comes from the plan.
    """
    MU, SD, PMU, PSD = (np.array(stats[k]) for k in ("MU", "SD", "PMU", "PSD"))
    Ps = [PL.frames(p)[:net.maxlen] for p in plans]
    Ts = [len(P) for P in Ps]
    B, T = len(Ps), max(Ts)
    Q = torch.zeros(B, T, 8, device=dev)
    pad = torch.ones(B, T, dtype=torch.bool, device=dev)
    x = torch.zeros(B, T, 9, device=dev)
    for i, P in enumerate(Ps):
        Q[i, :Ts[i]] = torch.tensor((P - PMU) / PSD, dtype=torch.float32, device=dev)
        pad[i, :Ts[i]] = False
        g = torch.Generator(device="cpu").manual_seed(int(seeds[i]) if seeds is not None else i)
        x[i, :Ts[i]] = torch.randn(Ts[i], 9, generator=g).to(dev)
    guided = cfg != 1.0
    has = torch.ones(B, 1, 1, device=dev)
    if guided:
        has, Q, pad = torch.cat([has, torch.zeros_like(has)]), torch.cat([Q, Q]), torch.cat([pad, pad])
    for k in range(steps):
        t = torch.full((len(has),), 1.0 - k / steps, device=dev)
        v = net(torch.cat([x, x]) if guided else x, t, Q, has, pad)
        if guided:
            v_cond, v_uncond = v[:B], v[B:]
            v = v_uncond + cfg * (v_cond - v_uncond)
        x = x - v / steps
    b, a = butter(4, lowpass_hz / (FPS / 2))
    out = []
    for i in range(B):
        A = x[i, :Ts[i]].cpu().numpy() * SD + MU
        if Ts[i] > 20 and lowpass_hz:
            A = filtfilt(b, a, A, axis=0, padlen=min(Ts[i] - 1, 15))
        out.append(A)
    return out


def generate(net, stats, plan, dev, seed=0, **kw):
    """One plan -> ``(T, 9)`` trajectory."""
    return generate_batch(net, stats, [plan], dev, seeds=[seed], **kw)[0]
