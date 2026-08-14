import pytest
import torch

from bitsandbytes import functional as F
from bitsandbytes.nn import Linear4bit
from bitsandbytes.nn.quant_refine import refine_4bit_weights

# Shapes that exercise the edge cases of the packing: a weight whose element
# count is not a multiple of the blocksize (zero-padded tail), and one that
# is exactly block-aligned.
SHAPES = [(37, 100), (32, 64)]


def _mse(weights: torch.Tensor, packed: torch.Tensor, state) -> float:
    dequantized = F.dequantize_4bit(packed, state)
    return ((weights.float() - dequantized.float()) ** 2).mean().item()


@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("blocksize", [32, 64, 128])
@pytest.mark.parametrize("shape", SHAPES)
def test_refine_4bit_weights_reduces_reconstruction_error(shape, blocksize, quant_type):
    torch.manual_seed(0)
    weights = torch.randn(*shape).to(torch.float16)

    packed, state = F.quantize_4bit(weights, blocksize=blocksize, quant_type=quant_type)
    before = _mse(weights, packed, state)

    refined_packed, refined_state = refine_4bit_weights(packed, state, weights, num_sweeps=3)
    after = _mse(weights, refined_packed, refined_state)

    # Refinement is accepted only when it strictly reduces the error, and the
    # round-to-nearest initializer is already code-optimal for the initial
    # scales, so a fresh quantization must improve or match exactly.
    assert after <= before + 1e-9


@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("shape", SHAPES)
def test_refine_4bit_weights_preserves_format(shape, quant_type):
    torch.manual_seed(0)
    weights = torch.randn(*shape).to(torch.float16)

    packed, state = F.quantize_4bit(weights, quant_type=quant_type)
    refined_packed, refined_state = refine_4bit_weights(packed, state, weights, num_sweeps=2)

    # The output must remain a drop-in replacement for the input: same packed
    # shape/dtype, same quantization parameters, and a dequantized tensor of
    # the original shape and dtype.
    assert refined_packed.shape == packed.shape
    assert refined_packed.dtype == packed.dtype
    assert refined_state.quant_type == state.quant_type
    assert refined_state.blocksize == state.blocksize
    assert refined_state.absmax.shape == state.absmax.shape
    dequantized = F.dequantize_4bit(refined_packed, refined_state)
    assert dequantized.shape == weights.shape
    assert dequantized.dtype == weights.dtype

    # The inputs are left untouched.
    original_absmax = state.absmax.clone()
    refine_4bit_weights(packed, state, weights, num_sweeps=2)
    assert torch.equal(state.absmax, original_absmax)


@pytest.mark.parametrize("quant_type", ["fp4", "nf4"])
@pytest.mark.parametrize("shape", SHAPES)
def test_refine_4bit_weights_rejects_nested_statistics(shape, quant_type):
    torch.manual_seed(0)
    weights = torch.randn(*shape).to(torch.float16)

    packed, state = F.quantize_4bit(weights, quant_type=quant_type, compress_statistics=True)
    with pytest.raises(ValueError, match="nested"):
        refine_4bit_weights(packed, state, weights)


def test_linear4bit_refines_when_enabled(monkeypatch):
    torch.manual_seed(0)
    weights = torch.randn(32, 64).to(torch.float16)

    def quantize():
        layer = Linear4bit(64, 32, compute_dtype=torch.float16, quant_type="nf4", compress_statistics=False)
        layer.weight.data = weights.t().clone()
        layer.to("cpu")
        return layer

    monkeypatch.setenv("BNB_REFINE_4BIT", "0")
    baseline = quantize()
    before = _mse(weights.t(), baseline.weight.data, baseline.weight.quant_state)

    monkeypatch.setenv("BNB_REFINE_4BIT", "3")
    refined = quantize()
    after = _mse(weights.t(), refined.weight.data, refined.weight.quant_state)

    assert baseline.weight.data.shape == refined.weight.data.shape
    assert after <= before + 1e-9


def test_linear4bit_default_is_unchanged(monkeypatch):
    monkeypatch.delenv("BNB_REFINE_4BIT", raising=False)
    torch.manual_seed(0)
    weights = torch.randn(32, 64).to(torch.float16)

    layer = Linear4bit(64, 32, compute_dtype=torch.float16, quant_type="nf4", compress_statistics=False)
    layer.weight.data = weights.t().clone()
    layer.to("cpu")

    # With the flag unset, the layer must behave exactly as before: the
    # scales are the plain absmax of each block, not a least-squares refit.
    quant_state = layer.weight.quant_state
    blocks = weights.t().float().reshape(-1, quant_state.blocksize)
    expected_absmax = blocks.abs().max(dim=1).values
    assert torch.allclose(quant_state.absmax.float(), expected_absmax, atol=1e-6)


def test_refine_4bit_weights_improves_on_worn_scales():
    """The refinement earns its keep when the initial scales are off.

    A round-to-nearest initializer is already code-optimal for absmax scales,
    so a fresh quantization has little room to move. Degenerate scales, by
    contrast, leave the codes far from optimal and refinement must recover a
    real share of the error.
    """
    torch.manual_seed(0)
    weights = torch.randn(32, 64).to(torch.float16)

    packed, state = F.quantize_4bit(weights, quant_type="nf4")
    # Inflate the scales so the assigned codes are systematically too small.
    state.absmax = state.absmax * 1.5
    degraded = _mse(weights, packed, state)

    refined_packed, refined_state = refine_4bit_weights(packed, state, weights, num_sweeps=3)
    recovered = _mse(weights, refined_packed, refined_state)

    assert refined_state.absmax.numel() == state.absmax.numel()
    assert recovered < degraded
