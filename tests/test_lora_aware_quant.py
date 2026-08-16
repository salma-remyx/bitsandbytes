import pytest
import torch

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit
from bitsandbytes.nn.lora_aware_quant import lora_aware_quantize_4bit
from bitsandbytes.nn.parametrize import replace_parameter_4bit
from tests.helpers import describe_dtype, get_available_devices, id_formatter, is_supported_on_hpu

REL_TOL = 1e-3


def relative_error(W, approx):
    return ((W.float() - approx.float()).norm() / W.float().norm()).item()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=describe_dtype)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("compress_statistics", [False], ids=id_formatter("compress_statistics"))
@pytest.mark.parametrize("blocksize", [64, 128])
@pytest.mark.parametrize("lora_rank", [4, 8])
def test_lora_aware_quantize_4bit_reduces_error(device, dtype, quant_type, compress_statistics, blocksize, lora_rank):
    """The joint (Q + BA) init must reconstruct W better than round-to-nearest alone."""
    if device == "hpu" and not is_supported_on_hpu(quant_type, dtype):
        pytest.skip("This configuration is not supported on HPU.")

    torch.manual_seed(0)
    W = torch.randn(32, 64, device=device, dtype=dtype)

    quantized, quant_state, lora_a, lora_b = lora_aware_quantize_4bit(
        W,
        lora_rank=lora_rank,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )

    # Same I/O contract as quantize_4bit: packed 4-bit payload plus a usable QuantState.
    assert quantized.dtype in (torch.uint8, torch.bfloat16, torch.float32)
    assert quant_state.quant_type == quant_type
    assert quant_state.blocksize == blocksize

    # Adapter shapes follow the LoRA convention for a [out, in] weight.
    assert lora_a.shape == (lora_rank, W.shape[1])
    assert lora_b.shape == (W.shape[0], lora_rank)
    assert lora_a.dtype == dtype and lora_b.dtype == dtype

    W_approx = bnb.functional.dequantize_4bit(quantized, quant_state) + (lora_b @ lora_a).float()
    plain_q, plain_state = bnb.functional.quantize_4bit(
        W, blocksize=blocksize, compress_statistics=compress_statistics, quant_type=quant_type
    )
    plain = bnb.functional.dequantize_4bit(plain_q, plain_state)
    assert relative_error(W, W_approx) < relative_error(W, plain)


def test_lora_aware_quantize_4bit_more_iterations_help():
    """Each alternating step should not blow up; iter=2 must stay a valid reconstruction."""
    torch.manual_seed(0)
    W = torch.randn(64, 64, dtype=torch.float32)

    _, _, a1, b1 = lora_aware_quantize_4bit(W, lora_rank=8, num_iter=1, quant_type="nf4")
    q2, qs2, a2, b2 = lora_aware_quantize_4bit(W, lora_rank=8, num_iter=2, quant_type="nf4")

    for a, b in ((a1, b1), (a2, b2)):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
    approx2 = bnb.functional.dequantize_4bit(q2, qs2) + (b2 @ a2)
    assert torch.isfinite(approx2).all()
    assert relative_error(W, approx2) < 0.5


def test_lora_aware_quantize_4bit_input_validation():
    W = torch.randn(8, 8)
    with pytest.raises(ValueError, match="2-D"):
        lora_aware_quantize_4bit(torch.randn(4, 8, 8), lora_rank=2)
    with pytest.raises(ValueError, match="lora_rank"):
        lora_aware_quantize_4bit(W, lora_rank=0)
    with pytest.raises(ValueError, match="num_iter"):
        lora_aware_quantize_4bit(W, lora_rank=2, num_iter=0)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_linear4bit_lora_rank_populates_adapters(device, quant_type):
    """The Linear4bit wiring: lora_rank must flow through Params4bit._quantize."""
    if device == "hpu" and not is_supported_on_hpu(quant_type):
        pytest.skip("This configuration is not supported on HPU.")

    torch.manual_seed(0)
    fp32 = torch.nn.Linear(64, 32, dtype=torch.float32, device=device)
    quantized = Linear4bit(
        64, 32, compute_dtype=torch.float32, quant_type=quant_type, device=device, lora_rank=8
    )
    quantized.load_state_dict(fp32.state_dict())
    quantized = quantized.to(device)

    assert quantized.weight.bnb_quantized
    assert getattr(quantized.weight, "lora_adapters", None) is not None
    lora_a, lora_b = quantized.weight.lora_adapters
    assert lora_a.shape == (8, 64)
    assert lora_b.shape == (32, 8)

    # Adapters actually close the gap the frozen quantized weight leaves open.
    W = fp32.weight.data
    deq = bnb.functional.dequantize_4bit(quantized.weight.data, quantized.weight.quant_state)
    assert relative_error(W, deq + lora_b @ lora_a) < relative_error(W, deq)

    # The layer is still a working Linear4bit.
    out = quantized(torch.randn(4, 64, device=device))
    assert out.shape == (4, 32)

    # Default path is unchanged: no adapters, no extra attribute.
    plain = Linear4bit(64, 32, compute_dtype=torch.float32, quant_type=quant_type, device=device)
    plain.load_state_dict(fp32.state_dict())
    plain = plain.to(device)
    assert getattr(plain.weight, "lora_adapters", None) is None
    assert not hasattr(plain.weight, "lora_rank")


@pytest.mark.parametrize("device", get_available_devices())
def test_replace_parameter_4bit_lora_rank_attaches_trainable_adapters(device):
    """The parametrize wiring: adapters land as trainable parameters beside the quantized one."""
    torch.manual_seed(0)
    module = torch.nn.Module()
    W = torch.randn(128, 64, device=device)
    module.weight = torch.nn.Parameter(W.clone())

    replace_parameter_4bit(module, "weight", quant_type="nf4", lora_rank=8)

    assert hasattr(module, "weight_lora_A") and hasattr(module, "weight_lora_B")
    assert module.weight_lora_A.requires_grad and module.weight_lora_B.requires_grad
    assert module.weight_lora_A.shape == (8, 64)
    assert module.weight_lora_B.shape == (128, 8)
    assert not module.weight.requires_grad

    W_approx = module.weight + (module.weight_lora_B @ module.weight_lora_A)
    assert relative_error(W, W_approx) < relative_error(W, module.weight)


def test_params4bit_lora_settings_survive_deepcopy():
    """LoRA-aware settings ride along with the parameter across copy/pickle (FSDP paths)."""
    import copy
    import pickle

    torch.manual_seed(0)
    fp32 = torch.nn.Linear(32, 16, dtype=torch.float32)
    quantized = Linear4bit(32, 16, compute_dtype=torch.float32, quant_type="nf4", lora_rank=4)
    quantized.load_state_dict(fp32.state_dict())
    quantized = quantized.to("cpu")
    assert quantized.weight.lora_rank == 4

    cloned = copy.deepcopy(quantized.weight)
    assert cloned.lora_rank == 4
    assert cloned.lora_num_iter == 1
    assert cloned.lora_adapters[0].shape == (4, 32)

    restored = pickle.loads(pickle.dumps(quantized.weight))
    assert restored.lora_rank == 4
    assert restored.lora_num_iter == 1

    # An ordinary Params4bit never grew the attribute and must stay that way.
    plain = Linear4bit(32, 16, compute_dtype=torch.float32, quant_type="nf4").to("cpu")
    assert getattr(plain.weight, "lora_rank", 0) == 0
    roundtripped = pickle.loads(pickle.dumps(plain.weight))
    assert roundtripped.lora_rank == 0


def test_params4bit_lora_adapters_survive_device_move():
    """A second `.to(device)` after quantization must not drop the adapters."""
    torch.manual_seed(0)
    fp32 = torch.nn.Linear(32, 16, dtype=torch.float32)
    quantized = Linear4bit(32, 16, compute_dtype=torch.float32, quant_type="nf4", lora_rank=4)
    quantized.load_state_dict(fp32.state_dict())
    quantized = quantized.to("cpu").to("cpu")

    assert getattr(quantized.weight, "lora_adapters", None) is not None
    lora_a, lora_b = quantized.weight.lora_adapters
    assert lora_a.shape == (4, 32) and lora_b.shape == (16, 4)
    assert torch.isfinite(lora_a).all() and torch.isfinite(lora_b).all()
