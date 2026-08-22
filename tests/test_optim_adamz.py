import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.optim.adamz import LossAdaptiveLrController, _ReferenceAdamZ
from tests.helpers import describe_dtype, get_available_devices, id_formatter

# Small enough that the paper's default patience=100 does not swallow the
# whole test horizon; the controller contract is independent of the values.
P = 10
S = 3


def test_controller_warmup_leaves_lr_untouched():
    controller = LossAdaptiveLrController(lr=1e-2, patience=P, stagnation_period=S)
    lr = 1e-2
    for loss in [1.0, 0.9, 5.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        lr = controller.observe_and_adjust(lr, loss)
    # The spike at 5.0 exceeds the past losses, but the history is shorter
    # than the patience window so no adjustment is made yet.
    assert lr == 1e-2


def test_controller_reduces_lr_on_overshoot():
    controller = LossAdaptiveLrController(lr=1e-2, patience=P, stagnation_period=S)
    lr = 1e-2
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        lr = controller.observe_and_adjust(lr, loss)
    # 5.0 is >= every loss in the patience window, including itself.
    lr = controller.observe_and_adjust(lr, 5.0)
    assert lr == pytest.approx(1e-2 * 0.5)


def test_controller_raises_lr_on_stagnation():
    controller = LossAdaptiveLrController(lr=1e-2, patience=P, stagnation_period=S)
    lr = 1e-2
    # A steadily decaying loss: the recent window collapses much faster than
    # the long one, which is the paper's plateau signature.
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25]:
        lr = controller.observe_and_adjust(lr, loss)
    assert lr == pytest.approx(1e-2 * 1.2)


def test_controller_clamps_lr_bounds():
    controller = LossAdaptiveLrController(
        lr=0.5,
        overshoot_factor=0.5,
        min_lr=0.4,
        max_lr=0.6,
        patience=P,
        stagnation_period=S,
    )
    lr = 0.5
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        lr = controller.observe_and_adjust(lr, loss)
    # An overshoot would take lr to 0.25; the floor holds it at min_lr.
    lr = controller.observe_and_adjust(lr, 5.0)
    assert lr == pytest.approx(0.4)


@pytest.mark.parametrize("gtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
def test_adamz_matches_reference_adam(gtype, device):
    """AdamZ must apply the same parameter update as its reference AdamZ and
    as plain Adam whenever the loss controller is not adjusting the rate --
    the paper's contribution is the rate schedule, not the update rule."""
    p1 = (torch.randn(1024, 32, device=device, dtype=gtype) * 0.1).float()
    p2 = p1.clone().to(device, gtype)
    p3 = p1.clone().to(device, gtype)

    reference = _ReferenceAdamZ([p1], lr=1e-3, patience=P, stagnation_period=S)
    adamz = bnb.optim.AdamZ([p2], lr=1e-3, patience=P, stagnation_period=S)
    plain_adam = bnb.optim.Adam([p3], lr=1e-3)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif gtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    else:
        atol, rtol = 1e-4, 1e-3

    for _ in range(5):
        g = (torch.randn(1024, 32, device=device, dtype=gtype) * 0.01).float()
        p1.grad = g.clone()
        p2.grad = g.clone().to(gtype)
        p3.grad = g.clone().to(gtype)

        # Only losses below the patience window are reported, so both AdamZ
        # variants hold the initial learning rate and must match plain Adam.
        reference.loss_tracker(0.5)
        adamz.loss_tracker(0.5)

        reference.step()
        adamz.step()
        plain_adam.step()

        torch.testing.assert_close(adamz.state[p2]["state1"], reference.state[p1]["exp_avg"], atol=atol, rtol=rtol)
        torch.testing.assert_close(adamz.state[p2]["state2"], reference.state[p1]["exp_avg_sq"], atol=atol, rtol=rtol)
        torch.testing.assert_close(p3.float(), p2.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
def test_adamz_overshoot_updates_parameters_with_reduced_lr(device):
    """Once the patience window is full, a loss spike must reach the shared
    update path: AdamZ steps with the shrunken rate while plain Adam keeps
    stepping with the scheduled one."""
    torch.manual_seed(0)
    p1 = torch.randn(1024, 32, device=device) * 0.1
    p2 = p1.clone()

    # Overshoot shrinks the rate before the update is applied.
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 5.0]
    adamz = bnb.optim.AdamZ([p2], lr=1e-2, patience=P, stagnation_period=S)
    plain_adam = bnb.optim.Adam([p1], lr=1e-2)

    for loss in losses:
        g = torch.randn(1024, 32, device=device) * 0.01
        p1.grad = g.clone()
        p2.grad = g.clone()

        adamz.loss_tracker(loss)
        adamz.step()
        plain_adam.step()

    assert adamz.param_groups[0]["lr"] == pytest.approx(1e-2 * 0.5)
    # The two trajectories started identical, so any divergence came from the
    # rate that reached the shared update path.
    assert (p2 - p1).abs().max() > 0


@pytest.mark.parametrize("device", get_available_devices(), ids=id_formatter("device"))
def test_adamz_registered_in_optim_namespace(device):
    """The 8-bit and paged variants must be constructible through the public
    `bnb.optim` namespace, like the rest of the optimizer family."""
    for cls in (bnb.optim.AdamZ, bnb.optim.AdamZ8bit, bnb.optim.PagedAdamZ, bnb.optim.PagedAdamZ8bit):
        p = torch.randn(1024, 32, device=device, requires_grad=True)
        optimizer = cls([p], lr=1e-3)
        p.grad = torch.randn(1024, 32, device=device) * 0.01
        optimizer.loss_tracker(0.5)
        optimizer.step()
        assert isinstance(optimizer, bnb.optim.AdamZ)
