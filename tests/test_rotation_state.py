import pytest
import torch

import bitsandbytes as bnb
from tests.helpers import get_available_devices, id_formatter
from bitsandbytes.optim.rotation_state import (
    BLOCKSIZE,
    LAMBDA_TO_DTYPE,
    angle_count,
    quantize_rotation,
    restore_rotation,
    rotation_bits,
)

# lam=1 (3.32-bit angles) is far too coarse to hold a first moment, so it is not
# covered here as a working setting; see test_rotation_state_gaussian_error.
LAMBDAS = [2, 3, 4]


@pytest.mark.parametrize("numel", [1, 2, 3, 255, 256, 257, 4096, 4097, 65537])
@pytest.mark.parametrize("lam", LAMBDAS)
def test_rotation_roundtrip_shape(numel, lam):
    torch.manual_seed(0)
    state = torch.randn(numel)

    theta, scale = quantize_rotation(state, lam)
    restored = restore_rotation(theta, scale, lam, numel)

    assert theta.dtype == LAMBDA_TO_DTYPE[lam]
    assert theta.numel() == angle_count(numel)
    assert scale.numel() == (angle_count(numel) + BLOCKSIZE - 1) // BLOCKSIZE
    assert restored.shape == state.shape
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("lam", LAMBDAS)
def test_rotation_state_gaussian_error(lam):
    """Relative round-trip error on a gaussian first moment stays small.

    lam=1 is excluded: at 3.32 bits per angle the reconstruction error is ~65%,
    which is not usable for an optimizer state.
    """
    torch.manual_seed(0)
    state = torch.randn(1 << 16) * 0.01

    theta, scale = quantize_rotation(state, lam)
    restored = restore_rotation(theta, scale, lam, state.numel())

    rel = ((restored - state).norm() / state.norm()).item()
    assert rel < 0.15


def test_rotation_bits_beats_8bit():
    """The hybrid layout must be cheaper than the 8-bit state path.

    The 8-bit path stores two states, each one byte per element plus a float32
    absmax per 256 values; `rotation_bits` is already summed over both states,
    so the reference here counts both too.
    """
    eight_bit_bits_per_elem = 2 * (8 + 32 / 256)
    assert rotation_bits(2) < eight_bit_bits_per_elem


def test_rotation_bits_reflects_dtype():
    """More angle precision costs more storage, and never exceeds 32-bit states."""
    assert rotation_bits(2) < rotation_bits(3) <= rotation_bits(4)
    assert rotation_bits(4) < 2 * 32


def test_rotation_rejects_bad_lambda():
    with pytest.raises(ValueError):
        quantize_rotation(torch.randn(8), 5)
    with pytest.raises(ValueError):
        restore_rotation(torch.zeros(4, dtype=torch.int16), torch.zeros(1), 5, 8)


@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
@pytest.mark.parametrize("lam", [2, 3])
def test_optimizer_rotation_state_layout(lam, device):
    """Selecting a rotation lambda rewrites the state layout via Optimizer2State."""
    p = torch.nn.Parameter(torch.randn(64, 64))
    p.grad = torch.randn_like(p)

    mng = bnb.optim.GlobalOptimManager.get_instance()
    mng.initialize()
    mng.override_config(p, "rotation_lam", lam)

    opt = bnb.optim.Adam([p], lr=1e-3, optim_bits=32)
    opt.step()

    state = opt.state[p]
    assert state["rotation_lam"] == lam
    # state1 is angle-coded at half the element count; state2 keeps the
    # 8-bit blockwise layout.
    assert state["state1"].dtype == LAMBDA_TO_DTYPE[lam]
    assert state["state1"].numel() == angle_count(p.numel())
    assert state["state2"].dtype == torch.uint8
    assert state["state2"].shape == p.shape
    assert torch.isfinite(p.data).all()


@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
@pytest.mark.parametrize("lam", [2, 3])
def test_optimizer_rotation_matches_32bit(lam, device):
    """Rotation-coded state1 must track the full-precision Adam trajectory.

    Only state1 is rotation-coded; state2 stays on the 8-bit blockwise path, so
    the reference is 8-bit Adam rather than 32-bit.
    """
    torch.manual_seed(0)
    p_ref = torch.nn.Parameter(torch.randn(64, 8))
    p_rot = torch.nn.Parameter(p_ref.data.clone())

    opt_ref = bnb.optim.Adam([p_ref], lr=1e-2, optim_bits=8)
    mng = bnb.optim.GlobalOptimManager.get_instance()
    mng.initialize()
    mng.override_config(p_rot, "rotation_lam", lam)
    opt_rot = bnb.optim.Adam([p_rot], lr=1e-2, optim_bits=32)

    grads = [torch.randn(64, 8) * 0.05 for _ in range(20)]
    for g in grads:
        p_ref.grad = g.clone()
        p_rot.grad = g.clone()
        opt_ref.step()
        opt_rot.step()

    assert torch.isfinite(p_rot.data).all()
    torch.testing.assert_close(p_rot.data, p_ref.data, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
@pytest.mark.parametrize("lam", [2, 3])
def test_optimizer_rotation_training(lam, device):
    """A full training loop converges to the same loss as 8-bit Adam."""
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(512, 32, generator=gen)
    W = torch.randn(32, 4, generator=gen) * 0.3
    Y = X @ W + 0.05 * torch.randn(512, 4, generator=gen)

    def final_loss(lam):
        torch.manual_seed(1)
        lin = torch.nn.Linear(32, 4)
        params = list(lin.parameters())
        if lam is not None:
            mng = bnb.optim.GlobalOptimManager.get_instance()
            mng.initialize()
            for p in params:
                mng.override_config(p, "rotation_lam", lam)
        opt = bnb.optim.Adam(params, lr=1e-2, optim_bits=32 if lam is not None else 8)
        for _ in range(200):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(lin(X), Y)
            loss.backward()
            opt.step()
        return loss.item()

    ref = final_loss(None)
    rot = final_loss(lam)

    assert torch.isfinite(torch.tensor(rot))
    assert rot < ref * 1.5


@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
def test_optimizer_without_rotation_unaffected(device):
    """With no rotation config the state layout is exactly as before."""
    p = torch.nn.Parameter(torch.randn(64, 64))
    p.grad = torch.randn_like(p)
    opt = bnb.optim.Adam([p], lr=1e-3, optim_bits=32)
    opt.step()
    state = opt.state[p]
    assert "rotation_lam" not in state
    assert state["state1"].dtype == torch.float32
