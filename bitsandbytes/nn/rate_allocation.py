# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Class-aware rate allocation for the softmax output head.

The lm_head of a small LLM holds 15-30% of all parameters, yet quantization
pipelines routinely leave it in high precision (``replace_linear`` skips it by
default). SoftWater (arXiv:2608.12026) poses head quantization as a
rate-distortion problem under the KL divergence between the original and the
quantized output distributions, and shows the optimal allocation is *class
aware*: the error on class ``k`` is weighted by that class' softmax curvature,
so frequent classes deserve a fine grid while rare classes can afford a coarse
one -- a large gap under Zipfian token distributions.

This module carries that allocation mechanism over the blockwise 4-bit grid
bitsandbytes already ships: the per-class bit rate is set by the *blocksize*
handed to :func:`bitsandbytes.functional.quantize_4bit`, row by row over the
vocabulary. Frequent rows get small blocks (more scales, lower error); rare rows
get large blocks (fewer scales, fewer stored bytes). No new kernel, no new
container format -- each row group is an ordinary 4-bit blockwise quantized
tensor, and the group sizes are the only extra state.

Adapted from "SoftWater: Class-Aware Rate Allocation for Softmax Quantization".
The paper's lattice encoder, its ``Kn x Kn`` Cholesky second-order weighting and
successive-interference-cancellation are out of scope here; the calibration-side
statistic survives as a token frequency prior, which the paper reports is the
signal that dominates the rate gap under Zipfian vocabularies.
"""

from typing import Optional

import torch

import bitsandbytes.functional as F

# Blockwise 4-bit scales are stored in float32, so the marginal cost of a block
# is (32 / blocksize) bits per weight against the 4 bits of payload.
SCALE_BITS = 32
PAYLOAD_BITS = 4

VALID_BLOCKSIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)

MIN_ZIPF_S = 1.0
MAX_ZIPF_S = 2.0


def zipf_frequencies(vocab_size: int, s: float = 1.07, device=None) -> torch.Tensor:
    """Unnormalized Zipf(a.k.a. Zeta) class probabilities over ranks 1..vocab_size.

    Stands in for the calibration pass when no token statistics are available:
    the paper's per-class curvature statistic is what makes frequency-aware
    allocation pay off, and a Zipf prior is its shape for natural language.

    Args:
        vocab_size (`int`): Number of classes (rows of the head).
        s (`float`, *optional*): Zipf exponent. Higher is more skewed.
        device: Device to place the returned tensor on.

    Returns:
        `torch.Tensor`: float32 probabilities summing to 1, most frequent first.
    """
    if not MIN_ZIPF_S <= s <= MAX_ZIPF_S:
        raise ValueError(f"s must be in [{MIN_ZIPF_S}, {MAX_ZIPF_S}], got {s}")
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=device)
    weights = ranks ** (-s)
    return weights / weights.sum()


def allocate_class_blocksizes(
    frequencies: torch.Tensor,
    in_features: int,
    target_bits: Optional[float] = None,
    coarse_blocksize: int = 2048,
    fine_blocksize: int = 32,
) -> torch.Tensor:
    """Assign a blocksize to every class row under a stored-bit budget.

    Rows are ranked by class frequency. The allocation is greedy over that
    ranking: spend the budget making the most frequent rows fine, and let the
    tail fall back toward ``coarse_blocksize``. This mirrors the paper's rate
    profile, which hands fine grids to frequent low-variance classes and coarse
    grids to rare ones.

    Args:
        frequencies (`torch.Tensor`): Per-class probabilities, shape ``[vocab_size]``.
        in_features (`int`): Row length of the head weight (``K``).
        target_bits (`float`, *optional*):
            Stored bits per weight across the whole head, scales included.
            Defaults to the uniform ``coarse_blocksize`` cost, i.e. "no finer
            than the baseline, but spend the savings where it counts".
        coarse_blocksize (`int`, *optional*): Grid for the rarest classes.
        fine_blocksize (`int`, *optional*): Grid for the most frequent classes.

    Returns:
        `torch.Tensor`: int64 blocksizes, shape ``[vocab_size]``.
    """
    for bound in (coarse_blocksize, fine_blocksize):
        if bound not in VALID_BLOCKSIZES:
            raise ValueError(f"blocksize {bound} not in {VALID_BLOCKSIZES}")
    if fine_blocksize > coarse_blocksize:
        raise ValueError(f"fine_blocksize {fine_blocksize} exceeds coarse_blocksize {coarse_blocksize}")

    vocab_size = frequencies.numel()
    if vocab_size == 0:
        return torch.empty(0, dtype=torch.int64, device=frequencies.device)

    def cost(bs: int) -> float:
        return PAYLOAD_BITS + SCALE_BITS / bs

    budget = cost(coarse_blocksize) if target_bits is None else float(target_bits)
    if budget < cost(coarse_blocksize):
        raise ValueError(
            f"target_bits {budget:.3f} is below the cheapest grid ({cost(coarse_blocksize):.3f} bits/weight)"
        )

    # Rank once, most frequent class first. The greedy pass then walks classes
    # in importance order and buys the finest grid that still fits the budget.
    order = torch.argsort(frequencies, descending=True)
    blocksizes = torch.full((vocab_size,), coarse_blocksize, dtype=torch.int64, device=frequencies.device)
    ladder = sorted((bs for bs in VALID_BLOCKSIZES if fine_blocksize <= bs <= coarse_blocksize), reverse=True)

    headroom = (budget - cost(coarse_blocksize)) * vocab_size * in_features
    for idx in order.tolist():
        current = blocksizes[idx].item()
        for candidate in ladder:
            if candidate >= current:
                continue  # not finer than what the row already has
            spend = (cost(candidate) - cost(current)) * in_features
            if spend <= headroom:
                headroom -= spend
                current = candidate
        blocksizes[idx] = current

    return blocksizes


class GroupedQuantState:
    """Quantization state for a class-grouped head: per-group :class:`F.QuantState`.

    ``payload_sizes`` records how many storage elements each group actually
    occupies in the packed buffer. It is captured at quantize time rather than
    recomputed, because :func:`bitsandbytes.functional.quantize_4bit` pads a
    group whose element count is odd, so the on-disk length is not always
    ``ceil(elements / 2)`` in a way a reader can reconstruct from ``shape`` alone.
    """

    def __init__(
        self,
        group_sizes: list[int],
        quant_states: list[F.QuantState],
        payload_sizes: list[int],
        shape,
        dtype,
        quant_type: str,
        storage_itemsize: int = 1,
    ):
        self.group_sizes = group_sizes
        self.quant_states = quant_states
        self.payload_sizes = payload_sizes
        self.shape = shape
        self.dtype = dtype
        self.quant_type = quant_type
        self.storage_itemsize = storage_itemsize

    @property
    def absmax(self) -> list[torch.Tensor]:
        return [qs.absmax for qs in self.quant_states]

    @property
    def blocksize(self) -> int:
        """Coarsest blocksize in use, for callers that only want a conservative value."""
        return self.quant_states[0].blocksize if self.quant_states else 64

    def nbytes_stored(self) -> int:
        """Stored bytes for payload plus float32 scales, which is the quantity the allocation budgets."""
        total = 0
        for qs, payload in zip(self.quant_states, self.payload_sizes):
            total += payload * self.storage_itemsize
            total += qs.absmax.numel() * qs.absmax.element_size()
        return total

    def to(self, device):
        self.quant_states = [qs.to(device) for qs in self.quant_states]
        return self


def quantize_head_classwise(
    weight: torch.Tensor,
    blocksizes: torch.Tensor,
    quant_type: str = "nf4",
    quant_storage: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, GroupedQuantState]:
    """Quantize a softmax head row-group by row-group, each group at its own blocksize.

    The input is ``[vocab_size, in_features]``, exactly as `torch.nn.Linear`
    holds a head weight, so the class axis SoftWater allocates over is simply
    the rows.

    Args:
        weight (`torch.Tensor`): Head weight in ``[vocab_size, in_features]`` orientation.
        blocksizes (`torch.Tensor`): Per-class blocksizes, shape ``[vocab_size]``.
        quant_type (`str`, *optional*): ``nf4`` or ``fp4``.
        quant_storage (`torch.dtype`, *optional*): Storage dtype for the packed payload.

    Returns:
        Tuple of the packed weight and its :class:`GroupedQuantState`.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D head weight, got shape {tuple(weight.shape)}")
    if blocksizes.numel() != weight.shape[0]:
        raise ValueError(
            f"blocksizes has {blocksizes.numel()} entries for {weight.shape[0]} classes; "
            "expected one per row of a [vocab_size, in_features] head weight"
        )

    device = weight.device
    packed_parts: list[torch.Tensor] = []
    group_sizes: list[int] = []
    quant_states: list[F.QuantState] = []
    payload_sizes: list[int] = []

    # Consecutive classes sharing a blocksize are merged into one group, so a
    # head whose allocation collapses to a single blocksize costs one call.
    bs_list = blocksizes.tolist()
    start = 0
    for end in range(1, len(bs_list) + 1):
        if end == len(bs_list) or bs_list[end] != bs_list[start]:
            group = weight[start:end].contiguous()  # [classes, in_features]
            packed, qs = F.quantize_4bit(
                group,
                blocksize=int(bs_list[start]),
                quant_type=quant_type,
                quant_storage=quant_storage,
            )
            packed_parts.append(packed.reshape(-1))
            group_sizes.append(end - start)
            quant_states.append(qs)
            payload_sizes.append(packed.numel())
            start = end

    packed = torch.cat(packed_parts) if packed_parts else torch.empty(0, device=device, dtype=quant_storage)
    state = GroupedQuantState(
        group_sizes=group_sizes,
        quant_states=quant_states,
        payload_sizes=payload_sizes,
        shape=tuple(weight.shape),
        dtype=weight.dtype,
        quant_type=quant_type,
        storage_itemsize=quant_storage_itemsize(quant_storage),
    )
    return packed, state


def quant_storage_itemsize(quant_storage: torch.dtype) -> int:
    return torch.empty(0, dtype=quant_storage).element_size()


def dequantize_head_classwise(packed: torch.Tensor, state: GroupedQuantState) -> torch.Tensor:
    """Rebuild the ``[vocab_size, in_features]`` head weight from its grouped 4-bit payload."""
    if packed.numel() != sum(state.payload_sizes):
        raise ValueError(
            f"packed buffer holds {packed.numel()} elements but the state accounts for "
            f"{sum(state.payload_sizes)}; wrong payload or truncated state"
        )

    rows = []
    offset = 0
    for qs, payload in zip(state.quant_states, state.payload_sizes):
        chunk = packed[offset : offset + payload].reshape(-1, 1)
        rows.append(F.dequantize_4bit(chunk, qs))
        offset += payload

    out = torch.cat(rows, dim=0).contiguous()  # back to [vocab_size, in_features]
    if out.shape != torch.Size(state.shape):
        raise ValueError(f"dequantized {tuple(out.shape)} does not match recorded {tuple(state.shape)}")
    return out.to(state.dtype).to(packed.device)


class ClassRateHeadLinear(torch.nn.Module):
    """A frozen softmax head kept in class-grouped 4-bit storage.

    Quantization happens on :meth:`from_head` from a high-precision head and a
    per-class frequency vector. The forward dequantizes group by group and does
    a plain linear, which keeps the head usable on any backend without a
    dedicated grouped kernel. That dequantize-per-forward is the cost of not
    having a grouped GEMM; it is the right trade for a research hook, and the
    natural place for a fused kernel to land later.

    Args:
        in_features (`int`): Row length of the head weight.
        out_features (`int`): Vocabulary size / number of classes.
        bias (`torch.Tensor`, *optional*): High-precision bias to carry over.
    """

    def __init__(self, in_features: int, out_features: int, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.packed_weight: Optional[torch.Tensor] = None
        self.quant_state: Optional[GroupedQuantState] = None
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", torch.zeros(out_features))

    @classmethod
    def from_head(
        cls,
        weight: torch.Tensor,
        blocksizes: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        quant_type: str = "nf4",
    ) -> "ClassRateHeadLinear":
        """``weight`` is ``[vocab_size, in_features]``, as `torch.nn.Linear` stores it."""
        head = cls(weight.shape[1], weight.shape[0], bias)
        packed, state = quantize_head_classwise(weight, blocksizes, quant_type=quant_type)
        head.packed_weight = packed
        head.quant_state = state
        return head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.packed_weight is None or self.quant_state is None:
            raise RuntimeError("ClassRateHeadLinear is not quantized; call from_head() first")
        w = dequantize_head_classwise(self.packed_weight, self.quant_state).to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias.to(x.dtype))


def class_rate_head_quantizer(
    in_features: int,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    frequencies: Optional[torch.Tensor] = None,
    zipf_s: float = 1.07,
    target_bits: Optional[float] = None,
    coarse_blocksize: int = 2048,
    fine_blocksize: int = 32,
    quant_type: str = "nf4",
) -> ClassRateHeadLinear:
    """Build a `ClassRateHeadLinear` for `bitsandbytes.utils.replace_linear`'s ``head_quantizer`` hook.

    Weights arrive as ``[vocab_size, in_features]``, so the class axis is the
    rows. Pass ``frequencies`` from calibration data over your deployment
    domain -- the paper's finding is that matching calibration to the deployment
    domain is what drives the rate gap, so a measured prior beats the Zipf
    default whenever you have one.
    """
    if weight.shape[1] != in_features:
        raise ValueError(f"weight shape {tuple(weight.shape)} does not match in_features {in_features}")

    if frequencies is None:
        frequencies = zipf_frequencies(weight.shape[0], s=zipf_s, device=weight.device)

    blocksizes = allocate_class_blocksizes(
        frequencies,
        in_features,
        target_bits=target_bits,
        coarse_blocksize=coarse_blocksize,
        fine_blocksize=fine_blocksize,
    )
    return ClassRateHeadLinear.from_head(weight, blocksizes, bias=bias, quant_type=quant_type)
