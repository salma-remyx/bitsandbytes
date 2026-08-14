"""
Fixed-grid refinement of already-quantized 4-bit weights.

Once a weight has been quantized, its integer codes are normally treated as
final. This module makes them improvable after the fact: starting from the
existing (codes, absmax) pair, it runs a backprop-free coordinate descent that
alternates between re-assigning each weight to its best 4-bit code on the
frozen quantization grid and re-fitting each block scale to the codes that
reference it. Updates are accepted only when they strictly reduce the mean
squared reconstruction error, and the result stays in the original quantized
format, so the refined weight dequantizes, saves and multiplies exactly like
the input did.

Adapted from ReQuant: Fixed-Grid Discrete Refinement for Post-Training
Quantization (https://arxiv.org/abs/2608.07019).
"""

from __future__ import annotations

import copy

import torch

from bitsandbytes.functional import QuantState

__all__ = ["refine_4bit_weights"]

# Minimum number of elements a block must contribute to its scale fit. Blocks
# whose codes are all zero (q @ q == 0) keep their existing absmax, since the
# least-squares scale is undefined for an all-zero code block.
_MIN_DENOMINATOR = 1e-12


def _unpack_4bit(packed: torch.Tensor, num_out: int) -> torch.Tensor:
    """Unpack `num_out` 4-bit codes from a uint8 tensor (two codes per byte).

    Follows bnb's packing convention: even-indexed values occupy the high
    nibble, odd-indexed values the low nibble.
    """
    flat = packed.reshape(-1).to(torch.uint8)
    codes = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=flat.device)
    codes[0::2] = flat >> 4
    codes[1::2] = flat & 0x0F
    return codes[:num_out]


def _pack_4bit(codes: torch.Tensor, num_out: int) -> torch.Tensor:
    """Pack 4-bit codes back into uint8, the inverse of `_unpack_4bit`."""
    num_bytes = (num_out + 1) // 2
    padded = torch.zeros(num_bytes * 2, dtype=torch.uint8, device=codes.device)
    padded[: codes.numel()] = codes
    packed = (padded[0::2] << 4) | padded[1::2]
    return packed[:num_bytes]


def _pad_to_blocks(flat: torch.Tensor, num_blocks: int, blocksize: int) -> torch.Tensor:
    """Right-pad a flat tensor to `num_blocks * blocksize` elements."""
    target = num_blocks * blocksize
    if flat.numel() == target:
        return flat
    padded = torch.zeros(target, dtype=flat.dtype, device=flat.device)
    padded[: flat.numel()] = flat
    return padded


def _best_codes(target: torch.Tensor, quant_map: torch.Tensor, blocksize: int) -> torch.Tensor:
    """Assign each element to the nearest grid value.

    `target` is a [num_blocks, blocksize] tensor already divided by the block
    scale, so the nearest grid value is also the optimal code in absolute terms.
    """
    # [num_blocks, blocksize, 16] distances to every grid point; memory use is
    # 4 * numel * 16 bytes in float32, chunked below for large weights.
    chunk_rows = max(1, min(target.shape[0], 2**24 // max(1, blocksize * 16)))
    best_codes = []
    for start in range(0, target.shape[0], chunk_rows):
        chunk = target[start : start + chunk_rows]
        distances = (chunk.unsqueeze(-1) - quant_map.view(1, 1, -1)).abs()
        best_codes.append(distances.argmin(-1).to(torch.uint8))
    return torch.cat(best_codes)


def refine_4bit_weights(
    packed_weights: torch.Tensor,
    quant_state: QuantState,
    original_weights: torch.Tensor,
    num_sweeps: int = 3,
) -> tuple[torch.Tensor, QuantState]:
    """Refine quantized 4-bit weights on their fixed grid.

    Runs `num_sweeps` rounds of coordinate descent over the existing codes and
    per-block scales, keeping the quantization grid itself frozen. Each sweep
    re-assigns every weight to its optimal code for the current scales, then
    re-fits each block scale to its new codes. Sweeps that would increase the
    reconstruction error are rolled back, so the returned weights are never
    worse than the input.

    Args:
        packed_weights (`torch.Tensor`): The packed 4-bit weights, as returned
            by `quantize_4bit`.
        quant_state ([`QuantState`]): The quantization state belonging to
            `packed_weights`. Must not use nested (double) quantization.
        original_weights (`torch.Tensor`): The unquantized weights that
            `packed_weights` was produced from. Only read, never modified.
        num_sweeps (`int`): Number of coordinate-descent sweeps to run.
            Defaults to 3.

    Returns:
        Tuple[`torch.Tensor`, `QuantState`]: The refined packed weights and a
        new `QuantState` carrying the refined scales. Both are new objects;
        the inputs are left untouched.
    """
    if quant_state.nested:
        raise ValueError(
            "refine_4bit_weights does not support nested (compressed) absmax statistics. "
            "Re-quantize with compress_statistics=False before refining."
        )

    blocksize = quant_state.blocksize
    num_out = quant_state.shape.numel()
    num_blocks = quant_state.absmax.numel()

    codes = _unpack_4bit(packed_weights, num_out)
    flat_original = original_weights.reshape(-1).float()
    padded_original = _pad_to_blocks(flat_original, num_blocks, blocksize).view(num_blocks, blocksize)

    quant_map = quant_state.code.to(device=codes.device, dtype=torch.float32)
    absmax = quant_state.absmax.float()

    def reconstruction_error(codes_flat: torch.Tensor, scales: torch.Tensor) -> float:
        grid_values = torch.zeros(num_blocks * blocksize, dtype=torch.float32, device=codes_flat.device)
        grid_values[: codes_flat.numel()] = quant_map[codes_flat.long()]
        values = grid_values.view(num_blocks, blocksize) * scales[:, None]
        # Only the first num_out positions carry real weights; the tail is padding.
        diff = (values - padded_original).reshape(-1)[:num_out]
        return (diff * diff).sum().item()

    def scaled_grid_codes(codes_flat: torch.Tensor) -> torch.Tensor:
        grid_values = torch.zeros(num_blocks * blocksize, dtype=torch.float32, device=codes_flat.device)
        grid_values[: codes_flat.numel()] = quant_map[codes_flat.long()]
        return grid_values.view(num_blocks, blocksize)

    best_error = reconstruction_error(codes, absmax)
    best_codes, best_absmax = codes, absmax

    for _ in range(num_sweeps):
        new_codes = _best_codes(padded_original / absmax[:, None], quant_map, blocksize).reshape(-1)
        q = scaled_grid_codes(new_codes)

        # Least-squares scale per block: absmax = argmin ||W - absmax * q||^2.
        numerator = (padded_original * q).sum(-1)
        denominator = (q * q).sum(-1)
        new_absmax = torch.where(
            denominator > _MIN_DENOMINATOR,
            numerator / denominator.clamp_min(_MIN_DENOMINATOR),
            absmax,
        )

        error = reconstruction_error(new_codes, new_absmax)
        if error >= best_error:
            break
        best_error, best_codes, best_absmax = error, new_codes, new_absmax

    refined_state = copy.copy(quant_state)
    refined_state.absmax = best_absmax.to(dtype=quant_state.absmax.dtype, device=quant_state.absmax.device)
    refined_packed = _pack_4bit(best_codes[:num_out], num_out).to(packed_weights.dtype)
    # Match the caller's packed layout ([num_bytes, 1] on CUDA, [1, num_bytes] in the
    # transposed BC path) so downstream dequantize_4bit sees the same convention.
    refined_packed = refined_packed.reshape(packed_weights.shape).contiguous()
    return refined_packed, refined_state
