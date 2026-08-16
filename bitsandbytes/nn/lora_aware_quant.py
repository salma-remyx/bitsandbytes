# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root of this source tree.
"""
LoRA-aware 4-bit quantization: jointly initialize a quantized weight and the low-rank
adapters that will be trained on top of it.

A plain QLoRA setup quantizes a pretrained weight ``W`` with round-to-nearest and then
attaches randomly-initialized LoRA adapters. The quantization error therefore sits
untouched in the frozen weight at the start of training, which is a large part of the
gap between QLoRA and full fine-tuning. Instead of quantizing ``W`` alone, we solve

    min over (Q, A, B) of ``|| W - (Q + B @ A) ||_F``  s.t. Q is 4-bit, rank(B @ A) <= r

by alternating between (1) quantizing the current target and (2) fitting a rank-r
truncated SVD to the residual. The returned quantized weight and ``QuantState`` are
drop-in for anything ``quantize_4bit`` produces; the adapters are the LoRA ``A``/``B``
to train instead of a random init.

Adapted from [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models]
(https://arxiv.org/abs/2310.08659). The alternating quantize/SVD scheme and the
joint objective are the paper's; the factorization is the symmetric ``sqrt(s)`` split
rather than the reference ``diag(s)`` split (identical product, better-conditioned
adapter norms), and the rank-r solve uses this repo's own ``quantize_4bit`` so the
output honors the existing NF4/FP4 blockwise ``QuantState`` contract.
"""

from typing import Optional

import torch

from bitsandbytes.functional import QuantState, dequantize_4bit, quantize_4bit


@torch.no_grad()
def lora_aware_quantize_4bit(
    W: torch.Tensor,
    lora_rank: int,
    num_iter: int = 1,
    blocksize: Optional[int] = None,
    compress_statistics: bool = False,
    quant_type: str = "fp4",
    quant_storage: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, QuantState, torch.Tensor, torch.Tensor]:
    """
    Quantize a 2-D weight to 4-bit while fitting LoRA adapters to the quantization residual.

    Args:
        W (`torch.Tensor`):
            The pretrained weight, shape `[out_features, in_features]`. Supports `float16`,
            `bfloat16`, or `float32`.
        lora_rank (`int`):
            Rank of the LoRA adapters fitted to the residual. Clamped to `min(W.shape) - 1`.
        num_iter (`int`, *optional*, defaults to `1`):
            Number of alternating quantize/SVD refinement steps. `1` matches the paper's
            default; each extra step costs one `quantize_4bit` plus one SVD.
        blocksize (`int`, *optional*, defaults to `None`):
            Block size for quantization. If `None`, uses the `quantize_4bit` default.
        compress_statistics (`bool`, *optional*, defaults to `False`):
            Whether to additionally quantize the absmax values.
        quant_type (`str`, *optional*, defaults to `"fp4"`):
            The quantization format to use: `nf4` or `fp4`.
        quant_storage (`torch.dtype`, *optional*, defaults to `torch.uint8`):
            The dtype of the tensor used to store the result.

    Returns:
        `tuple[torch.Tensor, QuantState, torch.Tensor, torch.Tensor]`:
        The packed 4-bit tensor, its `QuantState` (same contract as `quantize_4bit`), and
        the LoRA factors `lora_A` of shape `[lora_rank, in_features]` and `lora_B` of shape
        `[out_features, lora_rank]`, both in `W`'s dtype. Their product approximates the
        residual `W - dequantize_4bit(quantized, quant_state)`.

    Raises:
        ValueError: If `W` is not 2-D, or `lora_rank`/`num_iter` is not positive.
    """
    if W.dim() != 2:
        raise ValueError(f"lora_aware_quantize_4bit expects a 2-D weight, got shape {tuple(W.shape)}")
    if lora_rank < 1:
        raise ValueError(f"lora_rank must be >= 1, got {lora_rank}")
    if num_iter < 1:
        raise ValueError(f"num_iter must be >= 1, got {num_iter}")

    orig_dtype = W.dtype
    quant_kwargs = {
        "blocksize": blocksize,
        "compress_statistics": compress_statistics,
        "quant_type": quant_type,
        "quant_storage": quant_storage,
    }

    # Round-to-nearest on the unmodified weight is the t=0 iterate.
    Wf = W.detach().float()
    quantized, quant_state = quantize_4bit(W.detach(), **quant_kwargs)

    for _ in range(num_iter):
        residual = Wf - dequantize_4bit(quantized, quant_state).float()

        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        # Re-quantize what the adapters cannot represent, so the residual the *next*
        # round sees is the part of the error still left in the frozen weight.
        r = min(lora_rank, s.numel() - 1)
        sqrt_s = torch.sqrt(s[:r])
        lora_b = u[:, :r] * sqrt_s.unsqueeze(0)
        lora_a = sqrt_s.unsqueeze(1) * vh[:r, :]

        quantized, quant_state = quantize_4bit((Wf - lora_b @ lora_a).to(orig_dtype), **quant_kwargs)

    return quantized, quant_state, lora_a.to(orig_dtype), lora_b.to(orig_dtype)
