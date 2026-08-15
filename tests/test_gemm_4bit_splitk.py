import pytest
import torch

import bitsandbytes.functional as bnb_F
from bitsandbytes.backends.utils import _get_4bit_code
from bitsandbytes.backends.triton.ops import gemv_4bit


def _triton_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    pytest.skip("SplitK Triton kernel requires a Triton-capable device")


def _quantize_weight(N, K, quant_type, device, blocksize=64):
    torch.manual_seed(0)
    W = torch.randn(N, K, device=device) * 0.1
    packed, qs = bnb_F.quantize_4bit(W, blocksize=blocksize, quant_type=quant_type)
    return packed, qs.absmax


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("shape", [(64, 256), (128, 512), (32, 1024)])
@pytest.mark.parametrize("M", [1, 4])
def test_gemm_4bit_splitk_matches_dequantize_reference(quant_type, shape, M):
    """The fused SplitK GEMM must agree with dequantize_4bit + F.linear.

    This exercises the existing triton gemv_4bit wrapper (the op the xpu
    backend dispatches M==1 gemm_4bit calls to), which now routes through the
    fused kernel, against the default dequantize_4bit kernel as ground truth.
    """
    device = _triton_device()
    N, K = shape
    packed, absmax = _quantize_weight(N, K, quant_type, device)

    A = torch.randn(M, K, device=device, dtype=torch.float16) * 0.5

    code = _get_4bit_code(quant_type, torch.device(device))
    out_fused = gemv_4bit(A, packed, (N, K), absmax, code, 64)

    B_dq = torch.ops.bitsandbytes.dequantize_4bit.default(packed, absmax, 64, quant_type, (N, K), A.dtype)
    out_ref = torch.nn.functional.linear(A, B_dq)

    torch.testing.assert_close(out_fused, out_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_gemm_4bit_splitk_bias_and_splits(quant_type):
    """split_k>1 (partial reduce) and the bias add must both hold."""
    device = _triton_device()
    N, K = 96, 768
    packed, absmax = _quantize_weight(N, K, quant_type, device)

    A = torch.randn(2, K, device=device, dtype=torch.float16) * 0.5
    bias = torch.randn(N, device=device, dtype=torch.float16) * 0.1

    from bitsandbytes.backends.triton.gemm_splitk import gemm_4bit_splitk

    code = _get_4bit_code(quant_type, torch.device(device))
    out = gemm_4bit_splitk(A, packed, (N, K), absmax, code, 64, bias=bias, split_k=4)

    B_dq = torch.ops.bitsandbytes.dequantize_4bit.default(packed, absmax, 64, quant_type, (N, K), A.dtype)
    out_ref = torch.nn.functional.linear(A, B_dq, bias)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
