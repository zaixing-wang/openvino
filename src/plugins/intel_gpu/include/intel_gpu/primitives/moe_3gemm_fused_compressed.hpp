// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"

namespace cldnn {
using MOE3GemmFusedCompressed = ov::intel_gpu::op::MOE3GemmFusedCompressed;

struct moe_weights {
    cldnn::memory::ptr gate_w = nullptr;
    cldnn::memory::ptr gate_s = nullptr;
    cldnn::memory::ptr gate_z = nullptr;
    cldnn::memory::ptr up_w = nullptr;
    cldnn::memory::ptr up_s = nullptr;
    cldnn::memory::ptr up_z = nullptr;
    cldnn::memory::ptr down_w = nullptr;
    cldnn::memory::ptr down_s = nullptr;
    cldnn::memory::ptr down_z = nullptr;
};

/// @brief moe compressed primitive
/// @details Performs moe compressed
struct moe_3gemm_fused_compressed : public primitive_base<moe_3gemm_fused_compressed> {
    CLDNN_DECLARE_PRIMITIVE(moe_3gemm_fused_compressed)

    moe_3gemm_fused_compressed() : primitive_base("", {}) {}

    // @brief Constructs moe primitive / layer.
    //
    // @param id      An identifier of new primitive.
    // @param inputs  A list of Input primitive ids (inputs).
    //                   0: hidden_states - input tensor with hidden representations
    //                   1: routing_weights - [num_seq, num_experts] routing weights for all experts
    moe_3gemm_fused_compressed(const primitive_id& id, const std::vector<input_info>& inputs, const MOE3GemmFusedCompressed::Config& config, moe_weights weights, 
        const std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>& op, cldnn::memory::ptr base_mem)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          _config(config), _op(op) {
            _base = base_mem;
            _weights.gate_w = weights.gate_w;
            _weights.gate_s = weights.gate_s;
            _weights.gate_z = weights.gate_z;
            _weights.up_w = weights.up_w;
            _weights.up_s = weights.up_s;
            _weights.up_z = weights.up_z;
            _weights.down_w = weights.down_w;
            _weights.down_s = weights.down_s;
            _weights.down_z = weights.down_z;
          }

    MOE3GemmFusedCompressed::Config _config;
    std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed> _op;
    cldnn::memory::ptr _base;
    moe_weights _weights;
    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const moe_3gemm_fused_compressed>(rhs);

        return std::memcmp(&_config, &rhs_casted._config, sizeof(_config)) == 0;
    }

    void save(BinaryOutputBuffer& ob) const override {
        // std::cout << "wzx debug hit save" << std::endl;
        primitive_base<moe_3gemm_fused_compressed>::save(ob);
        ob << make_data(&_config, sizeof(_config));
        // ob << _weights.gate_w;
        // ob << _weights.gate_s;
        // ob << _weights.gate_z;
        // ob << _weights.up_w;
        // ob << _weights.up_s;
        // ob << _weights.up_z;
        // ob << _weights.down_w;
        // ob << _weights.down_s;
        // ob << _weights.down_z;
    }

    void load(BinaryInputBuffer& ib) override {
        // std::cout << "wzx debug hit load" << std::endl;
        primitive_base<moe_3gemm_fused_compressed>::load(ib);
        ib >> make_data(&_config, sizeof(_config));
        // ib >> _weights.gate_w;
        // ib >> _weights.gate_s;
        // ib >> _weights.gate_z;
        // ib >> _weights.up_w;
        // ib >> _weights.up_s;
        // ib >> _weights.up_z;
        // ib >> _weights.down_w;
        // ib >> _weights.down_s;
        // ib >> _weights.down_z;
        // create_weights_memory(ib.get_engine(), _weights.gate_w,_weights, _op);
    }
};

}  // namespace cldnn
