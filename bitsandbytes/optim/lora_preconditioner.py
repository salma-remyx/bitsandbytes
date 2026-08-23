# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Riemannian gradient preconditioning for LoRA factor pairs.

Implements the r x r preconditioner of Zhang & Pilanci, "Riemannian Preconditioned
LoRA for Fine-Tuning Foundation Models" (https://arxiv.org/abs/2402.02347). For a
LoRA layer written as ``delta_W = B @ A``, the euclidean gradient of each factor is
rescaled by the inverse Gram matrix of the *other* factor before it reaches the
optimizer:

    grad_A <- (B^T B + eps I)^-1 grad_A
    grad_B <- grad_B (A A^T + eps I)^-1

Both matrices are r x r with r the LoRA rank, so the rescale is cheap next to the
optimizer update. The rescaled gradients then flow through the unmodified update
step -- 32-bit, 8-bit blockwise and paged optimizers alike, since the moments are
estimated from the preconditioned gradient just as in the paper's AdamW variant.

Pairs are registered on the :class:`~bitsandbytes.optim.GlobalOptimManager` and
resolved against the optimizer's parameters at the first step. Attribute names may
be dotted paths, which is how the factors of a PEFT layer are reached:

    mng = bnb.optim.GlobalOptimManager.get_instance()
    for module in model.modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            mng.register_lora_pair(module, "lora_A.default.default_weight", "lora_B.default.default_weight")

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)

Unregistered parameters are never touched, so the preconditioner is strictly opt-in.
"""
import torch


def _resolve_attr(module, attr_path):
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_lora_pairs(module_pairs, param_groups):
    """Match registered LoRA pairs against an optimizer's parameters.

    Arguments:
        module_pairs (`list`):
            ``(module, attr_a, attr_b, eps)`` records as appended by
            ``GlobalOptimManager.register_lora_pair``.
        param_groups (`list`):
            The optimizer's param groups; only pairs whose factors are both found
            here are returned, so a registration never leaks into an unrelated
            optimizer.

    Returns:
        `list`: ``(lora_A, lora_B, eps)`` tuples of the resolved parameters.
    """
    param_ids = {id(p) for group in param_groups for p in group["params"]}
    pairs = []
    for module, attr_a, attr_b, eps in module_pairs:
        p_a = _resolve_attr(module, attr_a)
        p_b = _resolve_attr(module, attr_b)
        if id(p_a) in param_ids and id(p_b) in param_ids:
            pairs.append((p_a, p_b, eps))
    return pairs


def _check_pair(lora_a, lora_b):
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(
            f"LoRA factors must be 2D, got shapes {tuple(lora_a.shape)} and {tuple(lora_b.shape)}",
        )
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError(
            "LoRA factors do not share a rank r: lora_A is "
            f"{tuple(lora_a.shape)}, lora_B is {tuple(lora_b.shape)} "
            "(expected lora_A: (r, in_features), lora_B: (out_features, r) for delta_W = lora_B @ lora_A)"
        )
    return lora_a.shape[0]


@torch.no_grad()
def precondition_lora_grads(lora_pairs):
    """Rescale the gradients of LoRA factor pairs with the Riemannian preconditioner.

    Rewrites ``lora_A.grad`` and ``lora_B.grad`` in place (by rebinding) so the
    rescaled gradients are what the optimizer update sees. Each factor is
    preconditioned independently: the rescale of one factor needs only the weights
    of the other, so a factor with no gradient is simply skipped.

    The Gram matrices are formed and solved in float32 regardless of parameter
    dtype, since inverting them at half precision loses too many digits.

    Arguments:
        lora_pairs (`list`):
            ``(lora_A, lora_B, eps)`` tuples. ``eps`` is the damping added to the
            Gram diagonal, the paper's ``delta``; the reference implementation uses
            ``1e-6``.
    """
    for lora_a, lora_b, eps in lora_pairs:
        rank = _check_pair(lora_a, lora_b)
        eye = torch.eye(rank, device=lora_a.device)

        if lora_a.grad is not None:
            b = lora_b.float()
            grad = lora_a.grad
            lora_a.grad = torch.linalg.solve(b.T @ b + eps * eye, grad.float()).to(grad.dtype)

        if lora_b.grad is not None:
            a = lora_a.float()
            grad = lora_b.grad
            lora_b.grad = torch.linalg.solve(a @ a.T + eps * eye, grad.float().T).T.to(grad.dtype)
