# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Per-language low-rank corrections for already-quantized 4-bit layers.

Aggressive 4-bit quantization degrades non-English languages more than English.
This module implements the *library side* of a post-hoc remedy: a set of small,
per-language rank-``r`` additive corrections that are attached to the linear
layers of a model that has **already been quantized** with :class:`Linear4bit`,
and applied on top of the quantized matmul at dequantize time.

The quantized weights stay frozen and untouched. Each language owns its own
independent factors, and exactly one language is active at a time, so a single
quantized checkpoint can serve many languages by switching the active language
between forward passes -- no re-quantization, no model swap.

Adapted from "Language-Conditional Dequantization: Recovering What Quantization
Steals from Non-English Languages" (https://arxiv.org/abs/2608.11786). The
training loop that fits the factors against per-language calibration data is
deliberately out of scope here: it belongs in trainer code (PEFT-style), not in
the quantization library. What this module provides is the correction-aware
primitive the trainer attaches to.

Example:

```python
import torch
import bitsandbytes as bnb

model = ...  # already quantized with Linear4bit layers
attach_language_correction(model, languages=["en", "es", "de"], rank=2)

with bnb.nn.language_scope("es"):
    out = model(x)  # Spanish correction applied on top of the 4-bit matmul
```
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional, Union

import torch
from torch import nn

from .modules import Linear4bit

__all__ = [
    "LanguageCorrection",
    "attach_language_correction",
    "correction_parameter_fraction",
    "language_scope",
    "reset_active_language",
    "set_active_language",
]

# Ambient selection of which language's correction is active. A ``ContextVar``
# (rather than a plain module global) keeps concurrent forwards on separate
# threads / dataloader workers from clobbering each other's language choice.
_active_language: ContextVar[Optional[str]] = ContextVar("bnb_active_language", default=None)


def set_active_language(language: Optional[str]) -> object:
    """Select which language's correction subsequent forwards use.

    Args:
        language (`Optional[str]`): language code, or ``None`` to fall back to
            the plain quantized behavior (no correction applied).

    Returns:
        A token to pass to :func:`reset_active_language`.
    """
    return _active_language.set(language)


def reset_active_language(token: object) -> None:
    """Undo a :func:`set_active_language` call using its token."""
    _active_language.reset(token)


@contextmanager
def language_scope(language: Optional[str]) -> Iterator[None]:
    """Context manager that activates one language's corrections for its body."""
    token = set_active_language(language)
    try:
        yield
    finally:
        reset_active_language(token)


class _LowRankDelta(nn.Module):
    """One language's rank-``r`` additive correction for a single layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # B stays zero so that attaching a correction is an exact no-op until it
        # has been trained; A is drawn non-degenerate so gradients reach it
        # immediately once B moves off zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        with torch.no_grad():
            self.lora_B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the rank-2 inner dimension innermost: this is 2 GEMMs against a
        # thin factor rather than materializing an [out, in] delta.
        x = x.to(self.lora_A.dtype)
        return (x @ self.lora_A.T) @ self.lora_B.T


class LanguageCorrection(nn.Module):
    """Container holding one rank-``r`` correction per language for one layer.

    Args:
        in_features (`int`): input width of the layer being corrected.
        out_features (`int`): output width of the layer being corrected.
        languages (`list[str]`): language codes to create factors for.
        rank (`int`, *optional*, defaults to `2`): rank of each correction.
        dtype (`Optional[torch.dtype]`, *optional*): dtype of the factors.
            Should match the layer's compute dtype.
        device (`Optional[torch.device]`, *optional*): device of the factors.

    Calling it returns the additive delta for the currently active language, or
    ``None`` when no language is active or the active language has no factors --
    letting the caller skip the add entirely instead of adding zeros.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        languages: list[str],
        rank: int = 2,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        for language in languages:
            if not isinstance(language, str) or not language or "." in language:
                raise ValueError(f"invalid language code {language!r}")
        if len(set(languages)) != len(languages):
            raise ValueError(f"duplicate language codes in {languages}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.languages = list(languages)
        self.factors = nn.ModuleDict(
            {language: _LowRankDelta(in_features, out_features, rank, dtype, device) for language in languages}
        )

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, languages={self.languages}"

    @property
    def parameter_count(self) -> int:
        """Parameters added per language (all languages share this count)."""
        return self.rank * (self.in_features + self.out_features)

    @property
    def active_language(self) -> Optional[str]:
        """The language whose correction the next forward will apply."""
        language = _active_language.get()
        return language if language in self.factors else None

    def delta(self, language: Optional[str]) -> Optional[torch.Tensor]:
        """Materialize one language's full [out, in] weight delta, or ``None``."""
        if language is None or language not in self.factors:
            return None
        factor = self.factors[language]
        return (factor.lora_B @ factor.lora_A).to(factor.lora_B.dtype)

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        language = _active_language.get()
        if language is None or language not in self.factors:
            return None
        return self.factors[language](x)


def attach_language_correction(
    model: nn.Module,
    languages: list[str],
    rank: int = 2,
    dtype: Optional[torch.dtype] = None,
    skip_modules: Optional[list[str]] = None,
) -> list[str]:
    """Attach a :class:`LanguageCorrection` to every :class:`Linear4bit` in ``model``.

    The model must already be quantized and moved to its target device, so the
    corrections can be created on that device and in the right compute dtype.
    Layers are matched by class (``Linear4bit`` and subclasses), and the
    quantized weights are never modified. Attaching is idempotent: a layer that
    already carries a correction is left as-is.

    Args:
        model (`nn.Module`): quantized model to correct.
        languages (`list[str]`): language codes to create factors for.
        rank (`int`, *optional*, defaults to `2`): rank of each correction.
        dtype (`Optional[torch.dtype]`, *optional*): factor dtype; defaults to
            each layer's ``compute_dtype`` (falling back to ``float32``).
        skip_modules (`Optional[list[str]]`): module name suffixes to leave
            uncorrected, e.g. ``["lm_head"]``.

    Returns:
        `list[str]`: names of the modules a correction was attached to.

    Example:

    ```python
    quantized_model = quantized_model.to(0)
    attached = attach_language_correction(quantized_model, ["es", "de"])
    ```
    """
    skip = tuple(skip_modules or [])
    attached: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, Linear4bit):
            continue
        if skip and name.endswith(skip):
            continue
        if module.language_correction is not None:
            continue
        module.language_correction = LanguageCorrection(
            module.in_features,
            module.out_features,
            languages,
            rank=rank,
            dtype=dtype if dtype is not None else (module.compute_dtype or torch.float32),
            device=module.weight.device,
        )
        attached.append(name)
    return attached


def correction_parameter_fraction(model: nn.Module) -> dict[str, float]:
    """Report each language's correction size as a fraction of the corrected layers.

    The denominator is the *dequantized* weight element count of the layers that
    carry a correction -- the same basis on which a correction budget of ~0.1%
    of the model is normally quoted.

    Returns:
        `dict[str, float]`: language code -> fraction in [0, 1]. Empty when the
        model carries no corrections.
    """
    totals: dict[str, int] = {}
    base = 0
    for module in model.modules():
        correction = getattr(module, "language_correction", None)
        if not isinstance(correction, LanguageCorrection):
            continue
        base += module.in_features * module.out_features
        for language in correction.languages:
            totals[language] = totals.get(language, 0) + correction.parameter_count
    if base == 0:
        return {}
    return {language: count / base for language, count in sorted(totals.items())}
