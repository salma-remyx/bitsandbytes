"""Fused W4A16 GEMM (dequantize + matmul in one kernel) with SplitK decomposition.

Instead of materializing the full dequantized weight and calling a dense GEMM,
each program unpacks only the packed 4-bit nibbles it needs for its output tile,
scales them with the blockwise absmax, and feeds them straight into ``tl.dot``.
The K dimension is split across programs (SplitK) so skinny activations
(M small, K large — the dominant shape of decode-time Linear4bit calls) still
fill the device: every split computes a partial ``[M, N]`` tile in fp32 and the
host reduces the partials deterministically.

Adapted from "Accelerating a Triton Fused Kernel for W4A16 Quantized Inference
with SplitK work decomposition" (https://arxiv/abs/2402.00025). The paper's
fused dequant-in-GEMM and SplitK decomposition are kept at full fidelity; its
benchmark/autotuning sweep is left to downstream work (a fixed, conservative
tile/split heuristic is used here).
"""

from collections.abc import Sequence
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_4bit_splitk_kernel(
    a_ptr,
    b_ptr,
    code_ptr,
    absmax_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # int64 so N*K never overflows for embedding-scale weights
    flat = offs_n.to(tl.int64)[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    for k0 in range(k_start, k_end, BLOCK_K):
        k_offs = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None].to(tl.int64) * K + k_offs[None, :],
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < k_end),
            other=0.0,
        )

        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < k_end)
        byte = tl.load(b_ptr + (flat + k0) // 2, mask=w_mask, other=0)
        # element at even flat index lives in the high nibble, odd in the low
        nib = tl.where(((flat + k0) % 2) == 0, byte >> 4, byte & 0xF).to(tl.int32)
        w = tl.load(code_ptr + nib)
        scale = tl.load(absmax_ptr + (flat + k0) // BLOCKSIZE, mask=w_mask, other=0.0)

        acc += tl.dot(a, tl.trans((w * scale).to(a.dtype)))

    # partial [SPLIT_K, M, N] buffer; the host sums over axis 0
    c_ptrs = c_ptr + pid_k.to(tl.int64) * M * N + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _default_split_k(K: int) -> int:
    # keep each split's K-chunk in the ~512-element range, capped at 8 splits
    return max(1, min(8, K // 512))


def gemm_4bit_splitk(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
    bias: Optional[torch.Tensor] = None,
    split_k: Optional[int] = None,
) -> torch.Tensor:
    """A @ dequant(B).T + bias with in-kernel dequantization and SplitK over K.

    Args:
        A: activations ``[..., K]``.
        B: packed 4-bit weights ``[(N*K+1)//2, 1]`` (uint8 or a wider
            ``quant_storage`` dtype viewed as uint8).
        shapeB: original weight shape ``[N, K]``.
        absmax: float32 blockwise scales over the flattened ``[N, K]`` layout.
        code: 16-entry dequantization table for the quant type (nibble order).
        blocksize: quantization block size.
        bias: optional ``[N]`` bias.
        split_k: number of K splits; defaults to a shape-based heuristic.
    """
    N, K = int(shapeB[0]), int(shapeB[1])
    M = A.numel() // K
    orig_shape = A.shape

    if B.dtype != torch.uint8:
        B = B.squeeze().view(torch.uint8).unsqueeze(1)
    A = A.reshape(M, K)
    if not A.is_contiguous():
        A = A.contiguous()

    if split_k is None:
        split_k = _default_split_k(K)

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 64
    partials = torch.empty((split_k, M, N), dtype=torch.float32, device=A.device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), split_k)
    _gemm_4bit_splitk_kernel[grid](
        A,
        B,
        code,
        absmax,
        partials,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCKSIZE=blocksize,
        SPLIT_K=split_k,
        num_warps=4,
    )

    out = partials.sum(dim=0)
    if bias is not None:
        out = out + bias
    return out.to(A.dtype).reshape(*orig_shape[:-1], N)
