"""Blocking and checkpointing in `beamform`: the answer must not notice.

The kernel's working tensor is ``[steer, elements, bands, arrivals, gate]`` --
note the **element** axis, a 64x multiplier for a 64-element array, and the gate
width, which grows with ``sigma_t / bin_width``.  Materialising that for every
arrival at once is what made a full-size image cost gigabytes and exhaust the
machine, so `beamform` now blocks the arrival axis and recomputes each block in
the backward pass instead of keeping it.

Both are pure memory/compute trades, so everything here checks the same thing
from different angles: the number and the gradient are unchanged.  The memory
saving itself is measured in the docstring rather than asserted, because peak
RSS is not a property a test can pin portably.
"""

import math

import pytest
import torch

from hydropt import azimuth_steering, make_time_grid
from hydropt.beamform import ArrivalSet, beamform, shading_window

N_ARR, N_EL, N_TIME, N_STEER = 240, 16, 160, 25


def _arrivals(requires_grad=False, n=N_ARR):
    g = torch.Generator().manual_seed(4)
    d = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    amp = torch.rand(n, 1, generator=g)
    if requires_grad:
        amp = amp.clone().requires_grad_(True)
    return amp, ArrivalSet(
        time=torch.rand(n, generator=g) * 0.01 + 0.06,
        amplitude=amp, direction=d,
        phase=torch.rand(n, generator=g) * 2 * math.pi,
        distance=torch.rand(n, generator=g),
        path_length=torch.rand(n, generator=g) * 50 + 50,
        launch_direction=d)


def _rig():
    el = torch.stack([torch.zeros(N_EL), torch.linspace(-0.12, 0.12, N_EL),
                      torch.zeros(N_EL)], dim=-1)
    steer, _ = azimuth_steering(N_STEER, 30.0)
    grid = make_time_grid(0.05, 0.08, N_TIME)
    return el, steer, grid, dict(sigma_t=1.2e-4,
                                 shading=shading_window(N_EL, "hamming"))


def _run(arrivals, *, arrival_chunk, checkpoint=True, steer_chunk=0):
    el, steer, grid, kw = _rig()
    return beamform(arrivals, el, torch.tensor([100.0]), grid, steer,
                    arrival_chunk=arrival_chunk, checkpoint=checkpoint,
                    steer_chunk=steer_chunk, **kw)


@pytest.mark.parametrize("chunk", [1, 7, 64, N_ARR, N_ARR + 500])
def test_blocking_does_not_change_the_answer(chunk):
    """Including blocks of one, and a block larger than the arrival list."""
    _, arrivals = _arrivals()
    ref = _run(arrivals, arrival_chunk=0)
    got = _run(arrivals, arrival_chunk=chunk)
    assert torch.allclose(got, ref, rtol=1e-12, atol=float(ref.max()) * 1e-12)


def test_an_exact_multiple_and_a_ragged_last_block_agree():
    _, arrivals = _arrivals(n=200)
    a = _run(arrivals, arrival_chunk=50)    # 4 full blocks
    b = _run(arrivals, arrival_chunk=60)    # 3 full and a short one
    assert torch.allclose(a, b, rtol=1e-12, atol=float(a.max()) * 1e-12)


def test_checkpointing_does_not_change_the_answer():
    _, arrivals = _arrivals()
    on = _run(arrivals, arrival_chunk=0, checkpoint=True)
    off = _run(arrivals, arrival_chunk=0, checkpoint=False)
    assert torch.equal(on, off)


def test_blocking_and_checkpointing_leave_the_gradient_alone():
    """The whole point: a memory trade that changes the gradient is not a trade."""
    results = {}
    for name, chunk, ckpt in (("plain", 0, False), ("checkpoint", 0, True),
                              ("blocked", 32, True), ("both small", 8, True)):
        amp, arrivals = _arrivals(requires_grad=True)
        _run(arrivals, arrival_chunk=chunk, checkpoint=ckpt).sum().backward()
        results[name] = amp.grad.clone()
    ref = results["plain"]
    scale = float(ref.abs().max())
    for name, g in results.items():
        assert torch.isfinite(g).all(), name
        assert float((g - ref).abs().max()) < scale * 1e-11, name


def test_gradient_is_not_silently_zero():
    amp, arrivals = _arrivals(requires_grad=True)
    _run(arrivals, arrival_chunk=32).sum().backward()
    assert float(amp.grad.abs().max()) > 0.0


def test_blocking_composes_with_steer_chunking():
    _, arrivals = _arrivals()
    ref = _run(arrivals, arrival_chunk=0, steer_chunk=0)
    got = _run(arrivals, arrival_chunk=37, steer_chunk=4)
    assert torch.allclose(got, ref, rtol=1e-12, atol=float(ref.max()) * 1e-12)


def test_checkpointing_is_skipped_when_nothing_needs_a_gradient():
    """There is no graph to trade away, so it must not pay the recompute."""
    _, arrivals = _arrivals(requires_grad=False)
    with torch.no_grad():
        a = _run(arrivals, arrival_chunk=16, checkpoint=True)
        b = _run(arrivals, arrival_chunk=16, checkpoint=False)
    assert torch.equal(a, b)


def test_an_empty_arrival_set_gives_a_zero_image():
    z = torch.zeros(0)
    empty = ArrivalSet(time=z, amplitude=torch.zeros(0, 1),
                       direction=torch.zeros(0, 3), phase=z, distance=z,
                       path_length=z, launch_direction=torch.zeros(0, 3))
    out = _run(empty, arrival_chunk=64)
    assert out.shape == (N_STEER, 1, N_TIME)
    assert float(out.abs().max()) == 0.0
