"""Chunking must not change the answer.

Every ray is independent, so tracing a fan in chunks has to reproduce the
one-shot result, and an ETC accumulated over chunks has to reproduce the
one-shot ETC.  This is what makes the memory knobs (``ray_chunk``,
``render_chunked``, ``checkpoint_every``) safe to turn: they trade time for
memory and nothing else.
"""

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, MunkProfile, PiecewiseLinearProfile, Scene,
    make_time_grid, splat_etc, spherical_fan, vertical_line_array,
)


def _scene(**overrides) -> Scene:
    kw = dict(
        field=MunkProfile(),
        bottom=FlatHeight(1400.0),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, 1000.0),
        receivers=vertical_line_array(8000.0, 0.0, 600.0, 1400.0, 3),
        surface_loss=ConstantLoss(0.5),
        bottom_loss=ConstantLoss(4.0),
        freqs_khz=torch.tensor([0.2, 0.8]),
        step_size=40.0,
        n_steps=260,
    )
    kw.update(overrides)
    return Scene(**kw)


def _fan(n: int = 48) -> torch.Tensor:
    # Wide enough, in a 1400 m channel, that rays reaching the array have bounced
    # off both boundaries -- otherwise the boundary-loss gradients are correctly
    # zero and this file would not be testing them.
    return spherical_fan(n // 4, 4, (-25.0, 25.0), (-8.0, 8.0))


def _grid() -> torch.Tensor:
    return make_time_grid(5.0, 7.5, 200)


def test_tracing_in_chunks_reproduces_the_single_batch_paths():
    scene, dirs = _scene(), _fan()
    whole = scene.trace(dirs)
    parts = [scene.trace(dirs[lo:lo + 7]) for lo in range(0, dirs.shape[0], 7)]
    for field in ("pos", "tau", "arclen", "refl_db", "alive", "n_surface", "n_bottom"):
        joined = torch.cat([getattr(p, field) for p in parts], dim=0)
        # Rays never interact, so this is exact, not merely close.
        assert torch.equal(joined, getattr(whole, field)), f"{field} differs"


@pytest.mark.parametrize("ray_chunk", [1, 5, 16, 1000])
def test_splat_ray_chunking_reproduces_the_single_batch_etc(ray_chunk):
    scene, dirs, grid = _scene(), _fan(), _grid()
    result = scene.trace(dirs)
    kw = dict(sigma_d=150.0, sigma_t=8e-3)
    reference = splat_etc(result, scene.receivers, grid, scene.freqs_khz, **kw)
    chunked = splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                        ray_chunk=ray_chunk, **kw)
    # Only the summation order differs.
    assert torch.allclose(chunked, reference, rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize("chunk_size", [4, 11, 48])
def test_render_chunked_matches_render(chunk_size):
    scene, dirs, grid = _scene(), _fan(), _grid()
    reference = scene.render(dirs, grid, sigma_d=150.0, sigma_t=8e-3)
    chunked = scene.render_chunked(dirs, grid, chunk_size=chunk_size,
                                   sigma_d=150.0, sigma_t=8e-3)
    assert torch.allclose(chunked, reference, rtol=1e-12, atol=1e-30)
    assert reference.sum() > 0


@pytest.mark.parametrize("checkpoint_every", [1, 13, 260])
def test_checkpointing_does_not_change_the_forward_result(checkpoint_every):
    dirs, grid = _fan(), _grid()
    reference = _scene().render(dirs, grid, sigma_d=150.0, sigma_t=8e-3)
    ckpt = _scene(checkpoint_every=checkpoint_every).render(
        dirs, grid, sigma_d=150.0, sigma_t=8e-3)
    assert torch.equal(ckpt, reference)


def test_gradients_agree_between_chunked_and_single_batch():
    dirs, grid = _fan(), _grid()

    def grads(chunk_size: int | None):
        scene = _scene(
            field=PiecewiseLinearProfile([0.0, 700.0, 1400.0], [1520.0, 1500.0, 1530.0]),
            learn_source=True,
        )
        kw = dict(sigma_d=150.0, sigma_t=8e-3)
        etc = (scene.render(dirs, grid, **kw) if chunk_size is None
               else scene.render_chunked(dirs, grid, chunk_size=chunk_size, **kw))
        etc.sum().backward()
        return {n: p.grad.clone() for n, p in scene.named_parameters()}

    whole = grads(None)
    chunked = grads(9)
    assert set(whole) == set(chunked)
    for name in whole:
        assert whole[name].abs().sum() > 0, f"{name} gradient is identically zero"
        assert torch.allclose(chunked[name], whole[name], rtol=1e-10, atol=1e-20), name


def test_gradients_agree_with_and_without_checkpointing():
    dirs, grid = _fan(), _grid()

    def grads(checkpoint_every):
        scene = _scene(
            field=PiecewiseLinearProfile([0.0, 700.0, 1400.0], [1520.0, 1500.0, 1530.0]),
            learn_source=True, checkpoint_every=checkpoint_every,
        )
        scene.render(dirs, grid, sigma_d=150.0, sigma_t=8e-3).sum().backward()
        return {n: p.grad.clone() for n, p in scene.named_parameters()}

    plain, ckpt = grads(0), grads(17)
    for name in plain:
        assert plain[name].abs().sum() > 0, f"{name} gradient is identically zero"
        assert torch.allclose(ckpt[name], plain[name], rtol=1e-10, atol=1e-20), name


def test_receiver_array_is_handled_in_one_batched_call():
    """A whole array in one call must equal the same receivers rendered one at
    a time."""
    scene, dirs, grid = _scene(), _fan(), _grid()
    result = scene.trace(dirs)
    kw = dict(sigma_d=150.0, sigma_t=8e-3)
    together = splat_etc(result, scene.receivers, grid, scene.freqs_khz, **kw)
    for i in range(scene.receivers.shape[0]):
        alone = splat_etc(result, scene.receivers[i:i + 1], grid, scene.freqs_khz, **kw)
        assert torch.equal(alone[0], together[i])


def test_band_dimension_is_independent():
    """Rendering two bands together must equal rendering each alone: geometry is
    traced once and only the absorption exponent differs."""
    scene, dirs, grid = _scene(), _fan(), _grid()
    result = scene.trace(dirs)
    kw = dict(sigma_d=150.0, sigma_t=8e-3)
    both = splat_etc(result, scene.receivers, grid, scene.freqs_khz, **kw)
    for b in range(scene.freqs_khz.shape[0]):
        one = splat_etc(result, scene.receivers, grid, scene.freqs_khz[b:b + 1], **kw)
        assert torch.equal(one[:, 0], both[:, b])
