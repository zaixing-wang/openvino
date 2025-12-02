// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/moe_3gemm_fused_compressed_otd.hpp"

namespace ov::intel_gpu::op {
    std::shared_ptr<ov::Node> MOE3GemmFusedCompressedOTD::clone_with_new_inputs(const ov::OutputVector& new_args) const {
        check_new_args_count(this, new_args);
        return std::make_shared<MOE3GemmFusedCompressedOTD>(new_args);
    }
}  // namespace ov::intel_gpu::op