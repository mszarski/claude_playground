import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rmr import plan as PL
from rmr.generator.data import bucketize, fit_stats, samples_from_moves
from rmr.generator.model import MotionGenerator
from rmr.generator.sample import generate_batch
from rmr.generator.train import loss_fn
from rmr.motion import FPS, traj_to_move

TINY = dict(d=32, heads=2, layers=2, maxlen=200)


def _moves(n=3):
    out = []
    for k in range(n):
        t = np.arange(100 + 10 * k) / FPS
        A = np.zeros((len(t), 9))
        A[:, 4] = np.radians(10) * np.sin(2 * np.pi * (0.5 + 0.2 * k) * t)
        A[:, 6], A[:, 7] = -0.3 * np.sin(t), 0.3 * np.sin(t)
        out.append((f"clip{k}", traj_to_move(A)))
    return out


def test_default_model_size():
    n = sum(p.numel() for p in MotionGenerator().parameters())
    assert 21e6 < n < 22.5e6


def test_forward_shape_and_identity_init():
    net = MotionGenerator(**TINY)
    x = torch.randn(2, 50, 9)
    v = net(x, torch.rand(2), torch.randn(2, 50, 8), torch.ones(2, 1, 1), torch.zeros(2, 50, dtype=torch.bool))
    assert v.shape == x.shape
    assert torch.all(v == 0)          # zero-initialised output head


def test_data_pipeline_and_loss_decreases():
    torch.manual_seed(0)
    S = samples_from_moves(_moves())
    assert len(S) == 3 * 6            # mirror x 3 stretches
    stats = fit_stats(S)
    b = bucketize(S, stats, "cpu")[0]
    net = MotionGenerator(**TINY)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss = loss_fn(net, b["X"], b["Q"], b["M"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_generate_returns_plan_duration():
    S = samples_from_moves(_moves())
    net = MotionGenerator(**TINY).eval()
    plans = [{"duration": 2.0, "keys": [{"t": 0, "pitch": 0}, {"t": 2.0, "pitch": 10}]},
             {"duration": 3.0, "keys": [{"t": 0, "z": 5}]}]
    out = generate_batch(net, fit_stats(S), plans, "cpu", steps=2)
    assert [len(A) for A in out] == [50, 75]
    assert all(np.isfinite(A).all() and A.shape[1] == 9 for A in out)
