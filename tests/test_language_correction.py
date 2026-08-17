import pytest
import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, attach_language_correction, language_scope
from bitsandbytes.nn.language_correction import (
    LanguageCorrection,
    correction_parameter_fraction,
    reset_active_language,
    set_active_language,
)
from tests.helpers import describe_dtype, get_available_devices


def _quantized_model(device, dtype=torch.float32):
    """A small already-quantized model with two Linear4bit layers."""
    torch.manual_seed(0)
    return (
        nn.Sequential(
            Linear4bit(64, 64, bias=False, compute_dtype=dtype, quant_type="nf4"),
            nn.ReLU(),
            Linear4bit(64, 32, bias=False, compute_dtype=dtype, quant_type="nf4"),
        )
        .to(device)
        .eval()
    )


def _perturb(model, language):
    """Give one language's factors non-zero values in every corrected layer."""
    for module in model.modules():
        correction = getattr(module, "language_correction", None)
        if correction is not None:
            with torch.no_grad():
                correction.factors[language].lora_B.normal_(0.0, 0.1)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
def test_attach_is_noop_until_trained(device, dtype):
    if dtype == torch.float16 and device == "cpu":
        pytest.skip("float16 CPU matmul is unsupported.")

    model = _quantized_model(device, dtype)
    x = torch.randn(4, 64, device=device)
    before = model(x)

    attached = attach_language_correction(model, ["en", "es"], rank=2)
    assert attached == ["0", "2"]

    # No language selected -> identical output.
    assert torch.allclose(model(x), before, atol=1e-6 if dtype == torch.float32 else 1e-2)

    # Language selected but factors untrained (B == 0) -> still identical.
    with language_scope("es"):
        assert torch.allclose(model(x), before, atol=1e-6 if dtype == torch.float32 else 1e-2)


@pytest.mark.parametrize("device", get_available_devices())
def test_correction_is_language_conditional(device):
    model = _quantized_model(device)
    x = torch.randn(4, 64, device=device)
    before = model(x)

    attach_language_correction(model, ["en", "es"], rank=2)
    _perturb(model, "es")

    with language_scope("es"):
        corrected = model(x)
    assert not torch.allclose(corrected, before)

    # The other language's factors are untouched, so its output matches baseline.
    with language_scope("en"):
        assert torch.allclose(model(x), before)

    # The applied delta is exactly B @ A acting on the input, per layer.
    # Decompose: layer0 output = q0(x) + x @ d0.T; layer2 output = q2(relu(h)) + h @ d2.T,
    # where q_i is the plain quantized matmul (no language active).
    h0 = model[0](x)  # no language active -> plain quantized matmul
    d0 = model[0].language_correction.delta("es")
    h1 = torch.relu(h0 + x @ d0.T)
    d2 = model[2].language_correction.delta("es")
    expected = model[2](h1) + h1 @ d2.T
    assert torch.allclose(corrected, expected, atol=1e-5)


@pytest.mark.parametrize("device", get_available_devices())
def test_gradients_reach_only_active_language(device):
    model = _quantized_model(device)
    x = torch.randn(4, 64, device=device)

    attach_language_correction(model, ["en", "es"], rank=2)
    _perturb(model, "es")

    with language_scope("es"):
        model(x).square().mean().backward()

    factor = model[0].language_correction.factors["es"]
    assert factor.lora_B.grad is not None
    assert factor.lora_B.grad.abs().sum() > 0
    assert factor.lora_A.grad is not None

    # Inactive language gets no gradient.
    assert model[0].language_correction.factors["en"].lora_A.grad is None

    # The quantized base stays frozen.
    assert model[0].weight.requires_grad is False
    assert model[0].weight.grad is None


@pytest.mark.parametrize("device", get_available_devices())
def test_attach_skip_and_idempotent(device):
    model = nn.Sequential(
        Linear4bit(32, 32, bias=False, compute_dtype=torch.float32, quant_type="nf4"),
        nn.ReLU(),
        nn.Linear(32, 8),
    ).to(device)

    attached = attach_language_correction(model, ["es"], skip_modules=["0"])
    assert attached == []  # the only Linear4bit was skipped

    attached = attach_language_correction(model, ["es"])
    assert attached == ["0"]

    # Re-attaching neither duplicates nor replaces.
    assert attach_language_correction(model, ["es", "de"]) == []
    assert model[0].language_correction.languages == ["es"]


def test_parameter_fraction():
    model = _quantized_model("cpu")
    attach_language_correction(model, ["es"], rank=2)

    # rank * (in + out) per language over the dequantized weight elements.
    per_layer = 2 * (64 + 64) + 2 * (64 + 32)
    base = 64 * 64 + 64 * 32
    assert correction_parameter_fraction(model) == pytest.approx({"es": per_layer / base})

    assert correction_parameter_fraction(nn.Linear(4, 4)) == {}


def test_language_selection_restores():
    token = set_active_language("en")
    try:
        with language_scope("es"):
            with language_scope("de"):
                assert LanguageCorrection(8, 8, ["en", "es", "de"]).active_language == "de"
            assert LanguageCorrection(8, 8, ["en", "es", "de"]).active_language == "es"
        assert LanguageCorrection(8, 8, ["en", "es"]).active_language == "en"

        # An unknown language behaves like no correction at all.
        with language_scope("fr"):
            assert LanguageCorrection(8, 8, ["en"]).active_language is None
    finally:
        reset_active_language(token)


def test_state_dict_roundtrip_preserves_factors():
    model = _quantized_model("cpu")
    attach_language_correction(model, ["es"], rank=2)
    _perturb(model, "es")

    clone = _quantized_model("cpu")
    attach_language_correction(clone, ["es"], rank=2)
    clone.load_state_dict(model.state_dict(), strict=False)

    x = torch.randn(4, 64)
    with language_scope("es"):
        assert torch.allclose(clone(x), model(x), atol=1e-6)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="rank"):
        LanguageCorrection(8, 8, ["es"], rank=0)
    with pytest.raises(ValueError, match="language"):
        LanguageCorrection(8, 8, ["es", "es"])
    with pytest.raises(ValueError, match="language"):
        LanguageCorrection(8, 8, [""])


def test_delta_matches_factors():
    correction = LanguageCorrection(16, 8, ["es"], rank=3)
    with torch.no_grad():
        correction.factors["es"].lora_B.normal_(0.0, 0.5)

    assert correction.delta("en") is None
    delta = correction.delta("es")
    assert delta.shape == (8, 16)
    assert torch.allclose(delta, correction.factors["es"].lora_B @ correction.factors["es"].lora_A)

    # bnb.matmul_4bit stays reachable and unchanged for a plain layer.
    linear = Linear4bit(16, 8, bias=False, compute_dtype=torch.float32, quant_type="nf4").to("cpu")
    assert bnb.matmul_4bit(torch.randn(2, 16), linear.weight, quant_state=linear.weight.quant_state).shape == (2, 8)
