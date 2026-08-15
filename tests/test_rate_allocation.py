import functools

import pytest
import torch
from torch import nn

import bitsandbytes as bnb
from bitsandbytes.nn.rate_allocation import (
    PAYLOAD_BITS,
    SCALE_BITS,
    allocate_class_blocksizes,
    class_rate_head_quantizer,
    dequantize_head_classwise,
    quantize_head_classwise,
    zipf_frequencies,
)
from tests.helpers import get_available_devices, id_formatter


def stored_bits(blocksizes, in_features):
    return sum((PAYLOAD_BITS + SCALE_BITS / bs) * in_features for bs in blocksizes.tolist())


def relative_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("device", get_available_devices())
def test_allocation_is_ante_dependent_on_frequency(device):
    """Frequent classes must get the finer grid; the tail must absorb the coarseness."""
    vocab = 128
    frequencies = zipf_frequencies(vocab, s=1.2, device=device)

    blocksizes = allocate_class_blocksizes(
        frequencies, in_features=64, target_bits=4.5, coarse_blocksize=512, fine_blocksize=32
    )

    ranked = torch.argsort(frequencies, descending=True)
    assert blocksizes[ranked[0]] < blocksizes[ranked[-1]], "rarest class should not match the finest grid"

    # Monotone by rank: refining an earlier (more frequent) class never costs a later one.
    by_rank = blocksizes[ranked]
    assert torch.all(by_rank[:-1] <= by_rank[1:]), "allocation must be monotone in class frequency"


@pytest.mark.parametrize("device", get_available_devices())
def test_allocation_respects_the_rate_budget(device):
    vocab, in_features = 96, 64
    budget = 4.5

    blocksizes = allocate_class_blocksizes(
        zipf_frequencies(vocab, s=1.1, device=device),
        in_features=in_features,
        target_bits=budget,
        coarse_blocksize=512,
        fine_blocksize=32,
    )

    assert stored_bits(blocksizes, in_features) <= budget * vocab * in_features + 1e-6


def test_uniform_prior_collapses_to_the_coarse_grid():
    """With no frequency signal there is nothing to allocate, so nothing should be refined."""
    blocksizes = allocate_class_blocksizes(
        torch.full((32,), 1 / 32), in_features=64, coarse_blocksize=512, fine_blocksize=32
    )
    assert torch.all(blocksizes == 512)


def test_target_below_cheapest_grid_is_rejected():
    with pytest.raises(ValueError, match="below the cheapest grid"):
        allocate_class_blocksizes(zipf_frequencies(16), in_features=64, target_bits=4.0, coarse_blocksize=512)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("vocab", [64, 97], ids=id_formatter("vocab"))
def test_grouped_round_trip_shape_and_dtype(device, vocab):
    """Odd vocab sizes exercise the padding path in the underlying blockwise quantizer."""
    in_features = 96
    torch.manual_seed(0)
    weight = torch.randn(vocab, in_features, device=device).to(torch.float16)
    blocksizes = torch.full((vocab,), 2048, dtype=torch.int64, device=device)
    blocksizes[:17] = 32  # mixed, non-contiguous-along-the-ladder allocation

    packed, state = quantize_head_classwise(weight, blocksizes)

    assert packed.numel() == sum(state.payload_sizes)
    recovered = dequantize_head_classwise(packed, state)

    assert recovered.shape == weight.shape
    assert recovered.dtype == weight.dtype
    assert relative_error(recovered, weight) < 0.5


@pytest.mark.parametrize("device", get_available_devices())
def test_classwise_allocation_beats_uniform_at_matched_rate(device):
    """The point of the allocation: same stored bits, less distortion on the frequent head."""
    in_features, vocab = 128, 512
    torch.manual_seed(0)
    frequencies = zipf_frequencies(vocab, s=1.2, device=device)
    weight = (torch.randn(vocab, in_features, device=device) * frequencies.unsqueeze(1).sqrt()).to(torch.float16)

    target = 4.5
    classwise = allocate_class_blocksizes(
        frequencies, in_features=in_features, target_bits=target, coarse_blocksize=512, fine_blocksize=32
    )
    # A uniform grid whose stored cost matches: 4.5 bits/weight is blocksize 64.
    uniform = torch.full((vocab,), 64, dtype=torch.int64, device=device)

    assert abs(stored_bits(classwise, in_features) - stored_bits(uniform, in_features)) <= 1e-6 * vocab * in_features

    classwise_error = relative_error(
        dequantize_head_classwise(*quantize_head_classwise(weight, classwise)), weight
    )
    uniform_error = relative_error(dequantize_head_classwise(*quantize_head_classwise(weight, uniform)), weight)
    assert classwise_error < uniform_error


@pytest.mark.parametrize("device", get_available_devices())
def test_replace_linear_quantizes_the_head_instead_of_skipping_it(device):
    """Exercises the `head_quantizer` hook in `bitsandbytes.utils.replace_linear`."""
    in_features, hidden, vocab = 32, 32, 64
    torch.manual_seed(0)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(in_features, hidden)
            self.lm_head = nn.Linear(hidden, vocab)

        def forward(self, x):
            return self.lm_head(torch.nn.functional.relu(self.body(x)))

    model = TinyLM().to(device)
    reference = model(torch.randn(8, in_features, device=device))

    bnb.utils.replace_linear(
        model,
        lambda i, o, b: nn.Linear(i, o, b, device=device),
        head_quantizer=functools.partial(
            class_rate_head_quantizer, frequencies=zipf_frequencies(vocab, device=device), target_bits=4.5
        ),
    )

    assert isinstance(model.body, nn.Linear), "body must still go through linear_replacement"
    assert isinstance(model.lm_head, bnb.nn.ClassRateHeadLinear), "lm_head must be swapped by head_quantizer"

    logits = model(torch.randn(8, in_features, device=device))
    assert logits.shape == reference.shape

    quantized = model.lm_head.quant_state
    assert quantized is not None and quantized.quant_states, "head should carry grouped quant state"
    nbytes = quantized.nbytes_stored()
    assert nbytes < vocab * in_features * 2, "4-bit head must undercut the fp16 head it replaces"


@pytest.mark.parametrize("device", get_available_devices())
def test_head_linear_forward_tracks_the_dense_head(device):
    in_features, vocab = 64, 128
    torch.manual_seed(0)
    weight = torch.randn(vocab, in_features, device=device).to(torch.float16)
    bias = torch.randn(vocab, device=device).to(torch.float16)

    head = class_rate_head_quantizer(in_features, weight, bias=bias, target_bits=5.0)

    x = torch.randn(16, in_features, device=device).to(torch.float16)
    expected = nn.functional.linear(x, weight, bias)

    # 4-bit at ~5 stored bits/weight leaves a real footprint; assert the scale
    # of it rather than element-wise closeness.
    assert head(x).shape == expected.shape
    assert relative_error(head(x), expected) < 0.2


def test_head_quantizer_rejects_mismatched_in_features():
    weight = torch.randn(8, 16).to(torch.float16)  # [vocab, in_features]
    with pytest.raises(ValueError, match="does not match in_features"):
        class_rate_head_quantizer(4, weight)
