// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Matches Gather-based 2-GEMM MoE patterns (e.g. gpt-oss) and converts them
/// to MOECompressed with Expert_type::GEMM2_BIAS_SWIGLU_CLAMP.
///
/// The pattern is:
///   hidden → Gather(indices) → Unsqueeze → MatMul(gathered_weight) → Squeeze → Add(bias)
///   → SwiGLU activation (Slice/Clamp/Add + Slice/Minimum/Swish → Multiply)
///   → Unsqueeze → MatMul(gathered_down_weight) → Squeeze → Add(bias_down)
///   → Multiply(softmax_routing) → Reshape → ReduceSum
///
/// Weight decompression (Const(u8) → Convert → Subtract(zp) → Multiply(scale) → Convert)
/// is traced back to extract the raw compressed constant, scale, and zero point.
class ConvertGatheredMoeToMoeCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGatheredMoeToMoeCompressed");
    explicit ConvertGatheredMoeToMoeCompressed(bool has_batch_dim);
};

}  // namespace ov::intel_gpu
