# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from collections.abc import Callable, Iterable
from typing import Optional

import torch

from bitsandbytes.optim.optimizer import Optimizer2State


class LossAdaptiveLrController:
    """Adjusts a param-group learning rate from the observed loss trajectory.

    Two signals are tracked over a rolling loss history:

    * *Overshoot* -- the current loss is at least as large as every loss seen
      over the patience window. The step moved the parameters somewhere no
      better than anything in the recent past, so the learning rate is scaled
      down by ``overshoot_factor``.
    * *Stagnation* -- the spread of the loss over the (short) stagnation
      window has collapsed relative to the spread over the (long) patience
      window. Progress has plateaued, so the learning rate is scaled up by
      ``stagnation_factor``.

    The adjusted learning rate is clamped to ``[min_lr, max_lr]`` and written
    back to the param group, so any external scheduler that sets ``group["lr"]``
    keeps control between steps.
    """

    def __init__(
        self,
        lr: float,
        overshoot_factor: float = 0.5,
        stagnation_factor: float = 1.2,
        stagnation_threshold: float = 0.2,
        patience: int = 100,
        stagnation_period: int = 10,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
    ):
        if not 0.0 < overshoot_factor <= 1.0:
            raise ValueError(f"overshoot_factor must be in (0, 1], got {overshoot_factor}")
        if not stagnation_factor >= 1.0:
            raise ValueError(f"stagnation_factor must be >= 1.0, got {stagnation_factor}")
        if not 0.0 < stagnation_threshold:
            raise ValueError(f"stagnation_threshold must be > 0.0, got {stagnation_threshold}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        # The stagnation window must sit strictly inside the patience window,
        # otherwise its spread is compared against itself.
        if not 1 <= stagnation_period < patience:
            raise ValueError(f"stagnation_period must be in [1, patience), got {stagnation_period}")
        if not 0.0 <= min_lr <= lr <= max_lr:
            raise ValueError(f"Required min_lr <= lr <= max_lr, got {min_lr}, {lr}, {max_lr}")

        self.overshoot_factor = overshoot_factor
        self.stagnation_factor = stagnation_factor
        self.stagnation_threshold = stagnation_threshold
        self.patience = patience
        self.stagnation_period = stagnation_period
        self.min_lr = min_lr
        self.max_lr = max_lr

        # Losses are pushed newest-last; only the last `patience` are kept.
        self.loss_history: deque[float] = deque(maxlen=patience)

    def observe_and_adjust(self, lr: float, loss: float) -> float:
        """Record `loss` and return the adjusted learning rate for `lr`.

        The current loss is part of both windows: the paper tests
        ``L_t >= max({L_{t-p}, ..., L_t})`` and
        ``std({L_{t-s}, ..., L_t}) < sigma * std({L_{t-p}, ..., L_t})``.
        """
        self.loss_history.append(loss)
        if len(self.loss_history) < self.patience:
            # Not enough history for a fair comparison between the two windows.
            return lr

        new_lr = lr
        if loss >= max(self.loss_history):
            new_lr = lr * self.overshoot_factor
        elif self._is_stagnating():
            new_lr = lr * self.stagnation_factor

        return max(self.min_lr, min(new_lr, self.max_lr))

    def _is_stagnating(self) -> bool:
        recent = list(self.loss_history)[-self.stagnation_period :]
        long_term = list(self.loss_history)
        long_term_std = _std(long_term)
        if long_term_std == 0.0:
            # A flat long window carries no evidence of earlier progress, so
            # there is nothing to have stagnated away from.
            return False
        return _std(recent) < self.stagnation_threshold * long_term_std


def _std(values: list[float]) -> float:
    """Population standard deviation, matching the paper's use of `std`."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / n) ** 0.5


class _ReferenceAdamZ(torch.optim.Adam):
    """Reference implementation of AdamZ.

    AdamZ keeps the Adam update rule untouched and wraps it with a loss-driven
    learning-rate controller: the rate is shrunk when the loss overshoots the
    best of the recent patience window and grown when the loss plateaus.

    Reference: https://arxiv.org/abs/2411.15375
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        overshoot_factor: float = 0.5,
        stagnation_factor: float = 1.2,
        stagnation_threshold: float = 0.2,
        patience: int = 100,
        stagnation_period: int = 10,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.lr_controllers = [
            LossAdaptiveLrController(
                lr=group["lr"],
                overshoot_factor=overshoot_factor,
                stagnation_factor=stagnation_factor,
                stagnation_threshold=stagnation_threshold,
                patience=patience,
                stagnation_period=stagnation_period,
                min_lr=min_lr,
                max_lr=max_lr,
            )
            for group in self.param_groups
        ]

    @property
    def loss_tracker(self) -> Callable[[float], None]:
        """Register a callable that reports the training loss for each step.

        AdamZ reacts to the loss, which the optimizer step itself does not see;
        a user calls ``optimizer.loss_tracker(loss)`` (or
        ``optimizer.loss_tracker(float(loss.item()))``) once per step.
        """
        return self._record_loss

    def _record_loss(self, loss: float) -> None:
        for group, controller in zip(self.param_groups, self.lr_controllers, strict=True):
            group["lr"] = controller.observe_and_adjust(group["lr"], loss)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            self._record_loss(float(loss.item()))
        return super().step()


class AdamZ(Optimizer2State):
    """Adam with a loss-adaptive learning rate, following AdamZ.

    The Adam update itself is inherited unchanged from `Optimizer2State`;
    this class only rewrites ``group["lr"]`` before the update from the loss
    history, so the 32-bit and 8-bit blockwise state paths are shared with
    `Adam`.

    Arguments:
        params (`torch.Tensor`):
            The input parameters to optimize.
        lr (`float`, defaults to 1e-3):
            The learning rate.
        betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
            The decay rates of the first and second-order moment of the optimizer.
        eps (`float`, defaults to 1e-8):
            The epsilon value prevents division by zero in the optimizer.
        weight_decay (`float`, defaults to 0.0):
            The weight decay value for the optimizer.
        overshoot_factor (`float`, defaults to 0.5):
            Multiplier applied to the learning rate when the loss overshoots.
        stagnation_factor (`float`, defaults to 1.2):
            Multiplier applied to the learning rate when the loss stagnates.
        stagnation_threshold (`float`, defaults to 0.2):
            Fraction of the long-window loss spread below which the recent
            window counts as stagnating.
        patience (`int`, defaults to 100):
            Number of losses over which overshooting and the long-term spread
            are assessed.
        stagnation_period (`int`, defaults to 10):
            Number of recent losses the stagnation window covers.
        min_lr (`float`, defaults to 1e-7):
            Lower bound for the learning rate.
        max_lr (`float`, defaults to 1.0):
            Upper bound for the learning rate.
        optim_bits (`int`, defaults to 32):
            The number of bits of the optimizer state.
        min_8bit_size (`int`, defaults to 4096):
            The minimum number of elements of the parameter tensors for 8-bit optimization.
        is_paged (`bool`, defaults to `False`):
            Whether the optimizer is a paged optimizer or not.

    Reference: https://arxiv.org/abs/2411.15375
    """

    optimizer_name = "adam"

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        overshoot_factor: float = 0.5,
        stagnation_factor: float = 1.2,
        stagnation_threshold: float = 0.2,
        patience: int = 100,
        stagnation_period: int = 10,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        optim_bits: int = 32,
        args: Optional[dict] = None,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            "adam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits,
            args,
            min_8bit_size,
            is_paged=is_paged,
        )

        # One controller per param group. ``group["lr"]`` is read at every
        # update (see Optimizer2State.get_config), so writing it here is enough
        # to steer the shared 32-bit/8-bit update path.
        self.lr_controllers = [
            LossAdaptiveLrController(
                lr=group["lr"],
                overshoot_factor=overshoot_factor,
                stagnation_factor=stagnation_factor,
                stagnation_threshold=stagnation_threshold,
                patience=patience,
                stagnation_period=stagnation_period,
                min_lr=min_lr,
                max_lr=max_lr,
            )
            for group in self.param_groups
        ]

    @property
    def loss_tracker(self) -> Callable[[float], None]:
        """Register a callable that reports the training loss for each step.

        AdamZ reacts to the loss, which the optimizer step itself does not see;
        a user calls ``optimizer.loss_tracker(loss)`` once per step, before or
        after ``optimizer.step()``.
        """
        return self._record_loss

    def _record_loss(self, loss: float) -> None:
        for group, controller in zip(self.param_groups, self.lr_controllers, strict=True):
            group["lr"] = controller.observe_and_adjust(group["lr"], loss)


class AdamZ8bit(AdamZ):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        overshoot_factor: float = 0.5,
        stagnation_factor: float = 1.2,
        stagnation_threshold: float = 0.2,
        patience: int = 100,
        stagnation_period: int = 10,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            overshoot_factor=overshoot_factor,
            stagnation_factor=stagnation_factor,
            stagnation_threshold=stagnation_threshold,
            patience=patience,
            stagnation_period=stagnation_period,
            min_lr=min_lr,
            max_lr=max_lr,
            optim_bits=8,
            min_8bit_size=min_8bit_size,
            is_paged=is_paged,
        )


class PagedAdamZ(AdamZ):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        overshoot_factor: float = 0.5,
        stagnation_factor: float = 1.2,
        stagnation_threshold: float = 0.2,
        patience: int = 100,
        stagnation_period: int = 10,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        optim_bits: int = 32,
        min_8bit_size: int = 4096,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            overshoot_factor=overshoot_factor,
            stagnation_factor=stagnation_factor,
            stagnation_threshold=stagnation_threshold,
            patience=patience,
            stagnation_period=stagnation_period,
            min_lr=min_lr,
            max_lr=max_lr,
            optim_bits=optim_bits,
            min_8bit_size=min_8bit_size,
            is_paged=True,
        )


class PagedAdamZ8bit(AdamZ8bit):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        overshoot_factor: float = 0.5,
        stagnation_factor: float = 1.2,
        stagnation_threshold: float = 0.2,
        patience: int = 100,
        stagnation_period: int = 10,
        min_lr: float = 1e-7,
        max_lr: float = 1.0,
        min_8bit_size: int = 4096,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            overshoot_factor=overshoot_factor,
            stagnation_factor=stagnation_factor,
            stagnation_threshold=stagnation_threshold,
            patience=patience,
            stagnation_period=stagnation_period,
            min_lr=min_lr,
            max_lr=max_lr,
            min_8bit_size=min_8bit_size,
            is_paged=True,
        )
