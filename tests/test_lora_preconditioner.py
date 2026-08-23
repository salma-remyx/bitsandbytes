import pytest
import torch

import bitsandbytes as bnb
from tests.helpers import get_available_devices


def eye(rank, device):
    return torch.eye(rank, device=device)


class LoraLayer(torch.nn.Module):
    """delta_W = lora_B @ lora_A, the paper's parameterization."""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) * 0.05)
        self.lora_B = torch.nn.Parameter(torch.randn(out_features, rank) * 0.05)

    def delta(self, x):
        return x @ self.lora_A.T @ self.lora_B.T


def reference_precondition(lora_a, lora_b, grad_a, grad_b, eps=1e-6):
    """The paper's Eq. 1, spelled out: each factor's gradient is rescaled by the
    inverse (damped) Gram matrix of the other factor."""
    rank = lora_a.shape[0]
    eye = torch.eye(rank)
    precond_a = torch.linalg.inv(lora_b.T @ lora_b + eps * eye) @ grad_a
    precond_b = grad_b @ torch.linalg.inv(lora_a @ lora_a.T + eps * eye)
    return precond_a, precond_b


def test_register_lora_pair_is_opt_in():
    # without registration, an ordinary AdamW step must be untouched by the preconditioner
    layer = LoraLayer(16, 12, 4)
    opt = bnb.optim.AdamW(layer.parameters(), lr=1e-3)
    x = torch.randn(8, 16)
    layer.delta(x).sum().backward()

    grad_a, grad_b = layer.lora_A.grad.clone(), layer.lora_B.grad.clone()
    opt.step()

    # after one step the first moment is (1 - beta1) * grad
    state1 = opt.state[layer.lora_A]["state1"]
    torch.testing.assert_close(state1, 0.1 * grad_a, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.skipif(not get_available_devices(), reason="No device")
@pytest.mark.parametrize(
    "optim_factory",
    [
        bnb.optim.AdamW,
        bnb.optim.AdamW8bit,
        lambda p: bnb.optim.PagedAdamW8bit(p),
    ],
    ids=["adamw32", "adamw8bit", "paged_adamw8bit"],
)
def test_preconditioned_grad_reaches_optimizer(optim_factory, device):
    """The registered pair's gradients are rescaled before the update, so the first
    Adam moment is estimated from the preconditioned gradient, not the raw one."""
    torch.manual_seed(42)
    layer = LoraLayer(64, 48, 8).to(device)

    mng = bnb.optim.GlobalOptimManager.get_instance()
    mng.register_lora_pair(layer, "lora_A", "lora_B")

    opt = optim_factory(layer.parameters())
    x = torch.randn(4, 64, device=device)
    layer.delta(x).sum().backward()

    grad_a = layer.lora_A.grad.clone()
    grad_b = layer.lora_B.grad.clone()
    opt.step()

    assert len(opt.lora_pairs) == 1, "registered pair was not resolved against the optimizer"
    precond_a, _ = reference_precondition(layer.lora_A, layer.lora_B, grad_a, grad_b)

    state1 = opt.state[layer.lora_A]["state1"]
    # after one step the first moment is (1 - beta1) * precond_grad
    assert torch.equal(torch.sign(state1.float()), torch.sign(precond_a))
    torch.testing.assert_close(state1.float(), 0.1 * precond_a, rtol=0.3, atol=0.05)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.skipif(not get_available_devices(), reason="No device")
def test_precondition_matches_closed_form(device):
    torch.manual_seed(0)
    layer = LoraLayer(32, 24, 6).to(device)
    mng = bnb.optim.GlobalOptimManager.get_instance()
    mng.register_lora_pair(layer, "lora_A", "lora_B")

    # precondition a stale gradient directly, bypassing step()
    grad_a = torch.randn_like(layer.lora_A)
    grad_b = torch.randn_like(layer.lora_B)
    layer.lora_A.grad, layer.lora_B.grad = grad_a.clone(), grad_b.clone()

    bnb.optim.precondition_lora_grads([(layer.lora_A, layer.lora_B, 1e-6)])

    expected_a, expected_b = reference_precondition(layer.lora_A, layer.lora_B, grad_a, grad_b)
    torch.testing.assert_close(layer.lora_A.grad, expected_a, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(layer.lora_B.grad, expected_b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.skipif(not get_available_devices(), reason="No device")
def test_pair_across_optimizers_is_ignored(device):
    # a registration only takes effect if both factors are in the optimizer's params
    layer = LoraLayer(16, 12, 4).to(device)
    mng = bnb.optim.GlobalOptimManager.get_instance()
    mng.register_lora_pair(layer, "lora_A", "lora_B")

    opt = bnb.optim.AdamW([layer.lora_A], lr=1e-3)
    layer.lora_A.grad = torch.randn_like(layer.lora_A)
    opt.step()
    assert opt.lora_pairs == [], "a half pair leaked into an optimizer that only holds lora_A"


def test_mismatched_rank_is_rejected():
    a = torch.nn.Parameter(torch.randn(3, 8))
    b = torch.nn.Parameter(torch.randn(12, 4))
    a.grad, b.grad = torch.randn_like(a), torch.randn_like(b)
    with pytest.raises(ValueError, match="share a rank"):
        bnb.optim.precondition_lora_grads([(a, b, 1e-6)])


def test_dotted_attr_path():
    # PEFT nests the factor weights, e.g. lora_A.default.default_weight
    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            inner = LoraLayer(16, 12, 4)
            self.lora_A = torch.nn.Module()
            self.lora_A.default = torch.nn.Module()
            self.lora_A.default.default_weight = inner.lora_A
            self.lora_B = inner.lora_B

    nested = Nested()
    mng = bnb.optim.GlobalOptimManager.get_instance()
    mng.register_lora_pair(nested, "lora_A.default.default_weight", "lora_B")

    params = [nested.lora_A.default.default_weight, nested.lora_B]
    opt = bnb.optim.AdamW(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    assert len(opt.lora_pairs) == 1


def test_solve_matches_inverse_and_uses_gram_of_other_factor():
    # the preconditioner must use the *other* factor's Gram, not its own
    torch.manual_seed(1)
    layer = LoraLayer(20, 10, 5)
    grad_a = torch.randn_like(layer.lora_A)
    layer.lora_A.grad = grad_a.clone()
    layer.lora_B.grad = None  # B has no gradient: A is still preconditioned

    bnb.optim.precondition_lora_grads([(layer.lora_A, layer.lora_B, 1e-6)])

    own_gram = torch.linalg.inv(layer.lora_A @ layer.lora_A.T + 1e-6 * eye(5, layer.lora_A.device))
    assert not torch.allclose(layer.lora_A.grad, own_gram @ grad_a, rtol=0.1, atol=1e-3)
    other_gram = torch.linalg.inv(layer.lora_B.T @ layer.lora_B + 1e-6 * eye(5, layer.lora_A.device))
    torch.testing.assert_close(layer.lora_A.grad, other_gram @ grad_a, rtol=1e-4, atol=1e-5)
