# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .modules import (
    Embedding,
    Embedding4bit,
    Embedding8bit,
    EmbeddingFP4,
    EmbeddingNF4,
    Int8Params,
    Linear4bit,
    Linear8bitLt,
    LinearFP4,
    LinearNF4,
    OutlierAwareLinear,
    Params4bit,
    StableEmbedding,
)
from .rate_allocation import (
    ClassRateHeadLinear,
    GroupedQuantState,
    allocate_class_blocksizes,
    class_rate_head_quantizer,
    dequantize_head_classwise,
    quantize_head_classwise,
    zipf_frequencies,
)
