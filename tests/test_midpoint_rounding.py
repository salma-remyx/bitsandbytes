import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
from bitsandbytes.nn.midpoint_rounding import (
    apply_midpoint_rounding,
    midpoint_mask,
    refine_midpoint_rounding,
    select_midpoint_rounding,
)
from tests.helpers import get_available_devices


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_midpoint_rounding_reproduces_rtn_at_zero_tolerance(device, quant_type):
    """tolerance=0 must be bit-identical to the standard RTN path."""
    torch.manual_seed(0)
    weight = torch.randn(32, 64, device=device, dtype=torch.float32)
    rtn_packed, quant_state = bnb.functional.quantize_4bit(
        weight, quant_type=quant_type, compress_statistics=False
    )
    packed, _ = apply_midpoint_rounding(weight, quant_state, tolerance=0.0)
    assert torch.equal(packed.reshape(-1), rtn_packed.reshape(-1))


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_midpoint_rounding_changes_only_ambiguous_weights(device, quant_type):
    """Weights outside the ambiguous band must keep their RTN code index."""
    torch.manual_seed(1)
    weight = torch.randn(32, 64, device=device, dtype=torch.float32)
    _, quant_state = bnb.functional.quantize_4bit(weight, quant_type=quant_type, compress_statistics=False)

    code = quant_state.code.float()
    scale = quant_state.absmax.reshape(-1, 1)
    unit = weight.reshape(-1, quant_state.blocksize) / scale

    ref_packed, _ = apply_midpoint_rounding(weight, quant_state, tolerance=0.0)
    new_packed, _ = apply_midpoint_rounding(weight, quant_state, tolerance=0.2)
    mask = midpoint_mask(unit, code, 0.2).reshape(-1)

    ref_idx = torch.stack([ref_packed.reshape(-1) >> 4, ref_packed.reshape(-1) & 0xF], dim=1).reshape(-1)
    new_idx = torch.stack([new_packed.reshape(-1) >> 4, new_packed.reshape(-1) & 0xF], dim=1).reshape(-1)

    unambiguous = ~mask
    assert torch.equal(ref_idx[unambiguous], new_idx[unambiguous])
    # At least some ambiguous weights were re-rounded for this tolerance.
    assert not torch.equal(ref_idx[mask], new_idx[mask])


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_select_never_worse_than_rtn_spectrally(device, quant_type):
    """The selection step's guarantee: the chosen candidate's leading singular
    values are at least as close to the original's as plain RTN's."""
    torch.manual_seed(2)
    weight = torch.randn(96, 96, device=device, dtype=torch.float32)
    _, quant_state = bnb.functional.quantize_4bit(
        weight, quant_type=quant_type, compress_statistics=False
    )

    packed, state, tolerance = select_midpoint_rounding(weight, quant_state)
    selected = bnb.functional.dequantize_4bit(packed, state)
    rtn = bnb.functional.dequantize_4bit(*apply_midpoint_rounding(weight, quant_state, 0.0))

    top_k = 8
    s_ref = torch.linalg.svdvals(weight)
    err = lambda t: (torch.linalg.svdvals(t)[:top_k] - s_ref[:top_k]).abs().mean()
    assert err(selected) <= err(rtn) + 1e-6
    assert tolerance in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_refine_output_is_drop_in_for_dequantize(device, quant_type):
    """The refined packed tensor must round-trip through dequantize_4bit with
    the reused QuantState and reproduce the original shape/dtype."""
    torch.manual_seed(3)
    weight = torch.randn(64, 63, device=device, dtype=torch.float32)  # odd element count
    _, quant_state = bnb.functional.quantize_4bit(weight, quant_type=quant_type, compress_statistics=False)

    for tolerance in (0.05, "select"):
        packed, state = refine_midpoint_rounding(weight, quant_state, tolerance=tolerance)
        assert packed.dtype == torch.uint8
        dequantized = bnb.functional.dequantize_4bit(packed, state)
        assert dequantized.shape == weight.shape
        assert state is quant_state  # state reused unchanged


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_linear4bit_midpoint_tolerance_round_trip(device, quant_type):
    """End-to-end: Linear4bit(midpoint_tolerance=...) still quantizes, forwards,
    and stays bit-identical to the default layer when tolerance is 0."""
    torch.manual_seed(4)
    weight = torch.randn(32, 64, dtype=torch.float32)
    x = torch.randn(4, 64, device=device, dtype=torch.float32)

    def build(**kwargs):
        layer = Linear4bit(
            64,
            32,
            bias=False,
            quant_type=quant_type,
            compress_statistics=False,
            device="cpu",
            **kwargs,
        )
        with torch.no_grad():
            layer.weight.copy_(weight)
        return layer.to(device)

    default = build()
    assert default.weight.midpoint_tolerance is None

    zero = build(midpoint_tolerance=0.0)
    assert torch.equal(zero.weight.data.reshape(-1), default.weight.data.reshape(-1))
    assert torch.allclose(zero(x), default(x))

    refined = build(midpoint_tolerance=0.05)
    assert refined.weight.bnb_quantized
    out = refined(x)
    assert out.shape == (4, 32)
    # The re-rounded weights dequantize to valid grid points on the same scale.
    deq = bnb.functional.dequantize_4bit(refined.weight.data, refined.weight.quant_state)
    assert deq.shape == weight.shape
    assert not torch.equal(deq, bnb.functional.dequantize_4bit(default.weight.data, default.weight.quant_state))


def test_refine_rejects_bad_tolerance():
    weight = torch.randn(8, 64, dtype=torch.float32)
    _, quant_state = bnb.functional.quantize_4bit(weight, compress_statistics=False)
    with pytest.raises(ValueError, match="tolerance"):
        refine_midpoint_rounding(weight, quant_state, tolerance="nope")
