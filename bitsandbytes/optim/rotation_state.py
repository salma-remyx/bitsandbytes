# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Rotation-coded optimizer states.
#
# Adapted from "Irrational Complex Rotations Empower Low-bit Optimizers"
# (Tian et al., arXiv:2501.12896). A pair of state values (x, y) with
# x**2 + y**2 <= 4 is represented exactly as a single rotation angle
# theta through z = x + iy = e^{i*theta} + e^{i*pi_bar*theta}, where pi_bar is
# irrational. Lemma 3.2 of the paper gives theta in closed form, so the
# quantize/restore pair below is a linear-complexity elementwise computation
# with no search -- unlike the qmap binary search the 8-bit path performs.
#
# Deviation from the paper: Algorithm 1 scales by a single per-tensor
# maximum. Optimizer states are large and heavy-tailed, so a single scale
# wastes the dynamic range of most of the tensor. We keep the rotation
# encoding itself verbatim but apply it per block of paired values, which
# preserves the pair correlation while reusing bitsandbytes' blockwise
# scaling convention.
#
# The second deviation is which tensor gets coded at all. The paper codes both
# moments; here only the first moment is. The rotation map is dense near the
# origin, so values far below their block scale collapse toward zero, and the
# second moment is the one Adam divides by -- zeroing it makes the update
# unbounded. The first moment is roughly symmetric and survives the map well,
# so it is the half that carries the savings while state2 stays 8-bit.

import math

import torch

# Decimal digits of pi after "3.14159265", used to build pi_bar (paper Eq. 7).
_PI_TAIL = 0.35897932384626433832795028841971

# Supported lambda values and the integer dtype that holds theta (paper Eq. 10:
# theta_tilde = m * 10**lambda + g, so it needs 2*lambda decimal digits).
LAMBDA_TO_DTYPE = {1: torch.int8, 2: torch.int16, 3: torch.int32, 4: torch.int32}

# Number of paired values per scale block. Matches the 8-bit blockwise
# optimizer state path, which quantizes 256 values per absmax.
BLOCKSIZE = 256


def pibar(lam: int) -> float:
    """The irrational rotation coefficient for a given lambda (paper Eq. 7)."""
    return 10.0**-lam + 10.0 ** (-2 * lam) * _PI_TAIL


def rotation_bits(lam: int) -> float:
    """Bits per state element for the hybrid layout at this lambda.

    state1 is angle-coded: one angle of `LAMBDA_TO_DTYPE[lam]` per two
    elements, plus one float32 scale per block of pairs. state2 stays 8-bit
    blockwise: one byte per element plus one float32 absmax per 256 values.
    """
    numel = 2 * BLOCKSIZE  # a whole number of pair-blocks, so no padding
    angles = BLOCKSIZE
    state1_bits = angles * LAMBDA_TO_DTYPE[lam].itemsize * 8 + 32.0
    state2_bits = numel * 8 + 32.0
    return (state1_bits + state2_bits) / numel


def quantize_rotation(state: torch.Tensor, lam: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a float state tensor as rotation angles.

    The flattened tensor is split into (x, y) pairs of adjacent values and each
    pair is mapped to one quantized angle theta_tilde, halving the element
    count.

    Arguments:
        state (`torch.Tensor`):
            The float32 optimizer state to encode.
        lam (`int`):
            The decimal precision of the angle; 1..4, giving ~3.3 to ~13.3 bits
            per state element.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: the integer angle tensor of shape
        `(angle_count(numel),)` and the per-block float32 scales.
    """
    if lam not in LAMBDA_TO_DTYPE:
        raise ValueError(f"lam must be one of {sorted(LAMBDA_TO_DTYPE)}, got {lam}")

    flat = state.reshape(-1).float()
    numel = flat.numel()
    npair = (numel + 1) // 2
    nblocks = (npair + BLOCKSIZE - 1) // BLOCKSIZE
    # All blocks share one width so the angles pack into a single tensor; the
    # last block's tail is padding that is dropped on restore.
    per_block = (npair + nblocks - 1) // nblocks
    nangles = nblocks * per_block

    # x is the first npair values, y the rest. Pad each half out to nangles.
    x = flat[:npair]
    y = flat[npair:]
    y = torch.cat([y, y.new_zeros(npair - y.numel())])
    if nangles > npair:
        x = torch.cat([x, x.new_zeros(nangles - npair)])
        y = torch.cat([y, y.new_zeros(nangles - npair)])

    xb = x.view(nblocks, per_block)
    yb = y.view(nblocks, per_block)

    # Paper Eq. 9: scale each block into [-1, 1] so x**2 + y**2 <= 4.
    scale = torch.maximum(xb.abs().amax(dim=1), yb.abs().amax(dim=1))
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    xs, ys = xb / scale.unsqueeze(1), yb / scale.unsqueeze(1)

    # Paper Lemma 3.2 / Eq. 5-8.
    alpha = torch.atan2(ys, xs)
    beta = torch.acos(torch.clamp(torch.sqrt(xs * xs + ys * ys) / 2.0, -1.0, 1.0))
    pi_b = pibar(lam)
    omega = (alpha * (1.0 - pi_b) + beta * (1.0 + pi_b)) / (2.0 * math.pi)
    frac = omega - torch.floor(omega)
    m = torch.clamp(torch.floor(frac * 10.0**lam), 0, 10.0**lam - 1)

    # Paper Eq. 10.
    g = torch.floor((alpha - beta) / (2.0 * math.pi) * 10.0**lam)
    theta = m * 10.0**lam + g

    return theta.reshape(-1).to(LAMBDA_TO_DTYPE[lam]), scale


def restore_rotation(theta: torch.Tensor, scale: torch.Tensor, lam: int, numel: int) -> torch.Tensor:
    """Decode rotation angles back into a flat float state tensor.

    Arguments:
        theta (`torch.Tensor`):
            The integer angle tensor produced by [`quantize_rotation`].
        scale (`torch.Tensor`):
            The per-block scales produced by [`quantize_rotation`].
        lam (`int`):
            The lambda the angles were encoded with.
        numel (`int`):
            The element count of the original tensor, which is not recoverable
            from `theta` when it was odd or padded to a block boundary.

    Returns:
        `torch.Tensor`: the decoded flat state tensor.
    """
    if lam not in LAMBDA_TO_DTYPE:
        raise ValueError(f"lam must be one of {sorted(LAMBDA_TO_DTYPE)}, got {lam}")
    npair = (numel + 1) // 2
    if theta.numel() != angle_count(numel):
        raise ValueError(f"theta has {theta.numel()} angles for a {numel}-element state")

    # Paper Eq. 11.
    angle = theta.to(torch.float32) * (2.0 * math.pi * 10.0**-lam)
    pi_b = pibar(lam)
    xs = torch.cos(angle) + torch.cos(pi_b * angle)
    ys = torch.sin(angle) + torch.sin(pi_b * angle)

    nblocks = scale.numel()
    per_block = theta.numel() // nblocks
    x = (xs.view(nblocks, per_block) * scale.unsqueeze(1)).reshape(-1)
    y = (ys.view(nblocks, per_block) * scale.unsqueeze(1)).reshape(-1)

    return torch.cat([x[:npair], y[:npair]])[:numel]


def angle_count(numel: int) -> int:
    """The angle-tensor length for a state of `numel` elements."""
    npair = (numel + 1) // 2
    nblocks = (npair + BLOCKSIZE - 1) // BLOCKSIZE
    return nblocks * ((npair + nblocks - 1) // nblocks)


# The angle dtypes, for dispatch in Optimizer2State.update_step.
ROTATION_DTYPES = frozenset(LAMBDA_TO_DTYPE.values())


def init_rotation_state(state: dict, p: torch.Tensor, lam: int) -> None:
    """Allocate rotation-coded state1 and 8-bit blockwise state2 in `state`.

    Called from `Optimizer2State.init_state` when the parameter's config selects
    a rotation lambda. state1 becomes an angle tensor of half the parameter's
    element count plus per-block scales; state2 keeps the uint8/absmax layout
    the 8-bit path uses, so a state_dict round trip works unchanged.
    """
    state["rotation_lam"] = lam
    nangles = angle_count(p.numel())
    nblocks = (nangles + BLOCKSIZE - 1) // BLOCKSIZE
    dtype = LAMBDA_TO_DTYPE[lam]
    device = p.device

    state["state1"] = torch.zeros(nangles, dtype=dtype, device=device)
    state["scale1"] = torch.zeros(nblocks, dtype=torch.float32, device=device)

    # state2 stays on the 8-bit blockwise layout.
    n = p.numel()
    blocks = (n // 256) + bool(n % 256)
    state["state2"] = torch.zeros(n, dtype=torch.uint8, device=device)
    state["absmax2"] = torch.zeros(blocks, dtype=torch.float32, device=device)


@torch.no_grad()
def update_rotation_step(
    state: dict,
    grad: torch.Tensor,
    p: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    lr: float,
    weight_decay: float,
    qmap2: torch.Tensor = None,
) -> None:
    """One Adam-style step with state1 held in angle representation.

    Called from `Optimizer2State.update_step`. state1 (the first moment) is
    restored from angles, updated, and re-encoded (paper Algorithm 2). state2
    goes through the blockwise quantize/dequantize ops, reusing the 8-bit
    representation the repo already ships.

    Only state1 is rotation-coded. The rotation map is dense near the origin, so
    values far below their block scale collapse toward zero; on a heavy-tailed
    tensor that is more than a fifth of all elements. The first moment is
    roughly symmetric and well covered, but the second moment is what Adam
    divides by, and zeroing it there makes the update unbounded -- so it stays
    on the blockwise 8-bit path instead.
    """
    lam = state["rotation_lam"]
    numel = p.numel()

    state1 = restore_rotation(state["state1"], state["scale1"], lam, numel).view_as(p)
    # state2 reuses the 8-bit blockwise layout; go straight to the op the same
    # way the CPU optimizer kernel does.
    state2 = torch.ops.bitsandbytes.dequantize_blockwise.default(
        state["state2"], state["absmax2"], qmap2, 256, torch.float32
    ).view_as(p)

    grad = grad.float()
    state1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    state2.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    correction1 = 1.0 - beta1**step
    correction2 = math.sqrt(1.0 - beta2**step)

    p_data = p.data.float()
    if weight_decay > 0.0:
        p_data.mul_(1.0 - lr * weight_decay)
    p_data.addcdiv_(state1, (state2.sqrt() / correction2).add_(eps), value=-lr / correction1)
    p.data.copy_(p_data)

    state["state1"], state["scale1"] = quantize_rotation(state1, lam)
    state["state2"], state["absmax2"] = torch.ops.bitsandbytes.quantize_blockwise.default(state2, qmap2, 256)
