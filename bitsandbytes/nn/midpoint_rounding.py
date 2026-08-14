# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Midpoint-aware rounding for blockwise 4-bit quantization.

Adapted from ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity
in Calibration-Free LLM Quantization (https://arxiv.org/abs/2608.11045).

Round-to-nearest (RTN) is ill-conditioned for weights that fall close to the
midpoint between two adjacent quantization grid values: the rounding
direction is decided by differences smaller than the information the weight
carries. ReRound resolves this "midpoint ambiguity" in three steps:

1. a tolerance metric flags weights within a band around each grid midpoint,
2. only those weights are re-rounded from a guidance signal (weights near
   grid boundaries keep their RTN value),
3. the tolerance is swept to produce candidate quantized matrices, and the
   candidate whose leading singular values best match the original
   full-precision weights is selected.

This module implements that decision procedure on bnb's nf4/fp4 grid.
Steps 1 and 3 are ported as-is. For step 2 the paper trains a conditional
diffusion model per checkpoint to reconstruct the low-bit weights; that is
out of scope for an in-library default, so the guidance signal is a
parameter-free surrogate (an anti-RTN flip of the ambiguous weights),
callable-injectable for anyone who wants to plug a learned reconstruction
back in. Because tolerance 0 (pure RTN) is always in the sweep, the
selection step can only adopt a candidate that improves spectral fidelity
of the layer — it never makes the layer worse under its own metric.

Everything here is calibration-free and runs once, offline, at quantization
time; the packed uint8 tensor and QuantState contract is unchanged.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import torch

from bitsandbytes.functional import QuantState

logger = logging.getLogger(__name__)

# Sweep used when the caller asks ReRound's selection step to pick a tolerance.
DEFAULT_TOLERANCES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)

# `tolerance="select"` triggers the sweep + spectral selection described above.
Tolerance = Union[float, str]

# Quantization types whose grid is a fixed code table. Midpoint rounding only
# applies to these: the midpoint band is defined by adjacent table entries.
FIXED_GRID_QUANT_TYPES = ("nf4", "fp4")


def quant_type_uses_fixed_grid(quant_type: str) -> bool:
    """Whether `quant_type`'s grid is a fixed code table midpoint rounding applies to."""
    return quant_type in FIXED_GRID_QUANT_TYPES


def _nearest_code_index(values: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    """RTN indices: index of the nearest `code` entry for each flattened value.

    Exact-distance ties (the duplicated zeros in the fp4 table) are broken the
    way the quantization kernels break them: non-negative values take the
    upper-half index, negative values the lower-half one.
    """
    flat = values.reshape(-1, 1).float()
    dist = (flat - code.reshape(1, -1)).abs()
    half = code.numel() // 2
    # Nudge the sign-half the value does not belong to, so a tie resolves to
    # the kernel's choice without affecting any non-tied comparison.
    upper = (torch.arange(code.numel(), device=code.device) >= half).reshape(1, -1)
    prefers_upper = flat >= 0
    penalty = torch.where(prefers_upper, ~upper, upper).float() * 1e-6
    idx = (dist + penalty).argmin(dim=1)
    return idx.reshape(-1)


def midpoint_mask(
    values: torch.Tensor, code: torch.Tensor, tolerance: float
) -> torch.Tensor:
    """Mask of values inside the ambiguous band around a grid midpoint.

    A value is ambiguous when its distance to the nearer grid point is below
    ``tolerance * half_gap``, where half_gap is half the distance between the
    two grid points bracketing it. ``tolerance == 0`` masks nothing (pure
    RTN); ``tolerance >= 1`` masks every value that is not on a grid point.
    """
    v = values.reshape(-1, 1).float()
    lo = code[:-1].reshape(1, -1)  # lower bracketing grid point
    hi = code[1:].reshape(1, -1)  # upper bracketing grid point
    dist = torch.minimum((v - lo).abs(), (v - hi).abs())
    half_gap = 0.5 * (hi - lo)
    ambiguous = (dist < tolerance * half_gap) & (half_gap > 0)
    return ambiguous.any(dim=1).reshape(values.shape)


Guidance = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _anti_rtn_guidance(
    indices: torch.Tensor, values: torch.Tensor, code: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Default guidance signal: flip ambiguous weights to their other neighbor.

    "Neighbor" means the adjacent *grid value*, not the adjacent table index:
    the fp4 code table is listed in bit-pattern order, not value order, so the
    value-neighbor of index i is not generally i +/- 1. Since RTN already
    picked the nearer of the two bracketing grid points, this flip lands on
    the farther one — the opposite of what RTN chose. It is a stand-in for
    ReRound's diffusion-reconstructed weights: it has no knowledge of which
    direction is right, its role is to generate the alternative candidates
    that the spectral selection step then accepts or rejects.
    """
    val = values.reshape(-1).float()
    # The two nearest grid points — for an ambiguous weight these are the two
    # bracketing points.
    dist = (val.reshape(-1, 1) - code.reshape(1, -1)).abs()
    two_nearest = dist.topk(2, dim=1, largest=False).indices
    # The one of the pair that RTN did not pick.
    chosen = two_nearest == indices.reshape(-1, 1)
    other = torch.where(chosen[:, 0], two_nearest[:, 1], two_nearest[:, 0])
    out = indices.clone()
    m = mask.reshape(-1)
    out[m] = other[m]
    return out


def _block_scale(quant_state: QuantState) -> torch.Tensor:
    """Effective per-block absmax scale, undoing nested double quantization."""
    absmax = quant_state.absmax
    if quant_state.nested:
        from bitsandbytes.functional import dequantize_blockwise

        absmax = dequantize_blockwise(absmax, quant_state.state2) + quant_state.offset
    return absmax.reshape(-1, 1).float()


def _pack_indices(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Pack 4-bit indices two per byte, high nibble first (kernel layout)."""
    flat = indices.to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    return ((flat[0::2] << 4) | flat[1::2]).reshape(-1, 1)


def apply_midpoint_rounding(
    weight: torch.Tensor,
    quant_state: QuantState,
    tolerance: float,
    guidance: Optional[Guidance] = None,
) -> tuple[torch.Tensor, QuantState]:
    """Re-round weights in the ambiguous midpoint band of a 4-bit grid.

    Takes the *original* full-precision ``weight`` and the ``quant_state``
    from an RTN ``quantize_4bit`` pass, and returns a new packed tensor in
    which weights inside the ambiguous band were re-rounded by the guidance
    signal. The quant_state is reused unchanged, so the result is a drop-in
    replacement for the RTN packed tensor.

    Args:
        weight (`torch.Tensor`): The original full-precision weight tensor.
        quant_state ([`QuantState`]): State returned by `quantize_4bit` on `weight`.
        tolerance (`float`): Width of the ambiguous band around each grid
            midpoint, as a fraction of the half-gap between bracketing grid
            points. ``0`` reproduces plain RTN exactly.
        guidance (`Callable`, *optional*): Override for the default
            anti-RTN flip. Called as ``guidance(rtn_indices, unit_values,
            code, mask) -> indices`` — this is where a learned
            reconstruction (e.g. ReRound's diffusion model) would slot in.

    Returns:
        Tuple[`torch.Tensor`, `QuantState`]: Re-rounded packed weights and
        the unchanged quantization state.
    """
    code = quant_state.code.float()
    blocksize = quant_state.blocksize

    unit = weight.reshape(-1, blocksize).float() / _block_scale(quant_state)
    idx = _nearest_code_index(unit, code)
    if tolerance > 0:
        mask = midpoint_mask(unit, code, tolerance)
        guide = guidance if guidance is not None else _anti_rtn_guidance
        idx = guide(idx, unit, code, mask)
        logger.debug("midpoint rounding: re-rounded %d/%d weights", int(mask.sum()), idx.numel())

    return _pack_indices(idx, weight.device), quant_state


def _spectral_distance(candidate: torch.Tensor, reference: torch.Tensor, top_k: int) -> float:
    """Mean absolute error between the `top_k` leading singular values."""
    k = min(top_k, min(candidate.shape))
    s_ref = torch.linalg.svdvals(reference.float())[:k]
    s_cand = torch.linalg.svdvals(candidate.float())[:k]
    return (s_cand - s_ref).abs().mean().item()


def select_midpoint_rounding(
    weight: torch.Tensor,
    quant_state: QuantState,
    tolerances: tuple[float, ...] = DEFAULT_TOLERANCES,
    guidance: Optional[Guidance] = None,
    top_k: int = 8,
) -> tuple[torch.Tensor, QuantState, float]:
    """Sweep the tolerance and select the candidate with the best spectral match.

    This is ReRound's selection step: each tolerance yields a candidate
    quantized matrix; the returned candidate is the one whose leading
    singular values are closest to the original full-precision weights.
    ``0.0`` is always part of the sweep, so the selection never returns a
    candidate that is worse than plain RTN under this metric.

    Returns the selected ``(packed_tensor, quant_state, tolerance)``.
    """
    from bitsandbytes.functional import dequantize_4bit

    best = None
    for tau in sorted(set(tolerances) | {0.0}):
        packed, state = apply_midpoint_rounding(weight, quant_state, tau, guidance)
        dequantized = dequantize_4bit(packed, state)
        if dequantized.shape != weight.shape:
            dequantized = dequantized.reshape(weight.shape)
        score = _spectral_distance(dequantized, weight, top_k)
        if best is None or score < best[0]:
            best = (score, packed, state, tau)
    logger.debug("midpoint rounding: selected tolerance %.3g (spectral distance %.5f)", best[3], best[0])
    return best[1], best[2], best[3]


def refine_midpoint_rounding(
    weight: torch.Tensor,
    quant_state: QuantState,
    tolerance: Tolerance = "select",
    guidance: Optional[Guidance] = None,
) -> tuple[torch.Tensor, QuantState]:
    """Entry point used by `Params4bit._quantize`.

    ``tolerance`` is either a float band width passed straight through to
    `apply_midpoint_rounding`, or ``"select"`` to run the sweep + spectral
    selection of `select_midpoint_rounding`.
    """
    if isinstance(tolerance, str):
        if tolerance != "select":
            raise ValueError(f"tolerance must be a float or 'select', got {tolerance!r}")
        packed, state, _ = select_midpoint_rounding(weight, quant_state, guidance=guidance)
        return packed, state
    return apply_midpoint_rounding(weight, quant_state, tolerance, guidance)
