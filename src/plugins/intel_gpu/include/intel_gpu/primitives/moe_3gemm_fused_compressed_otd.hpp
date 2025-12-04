// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/op/moe_3gemm_fused_compressed_otd.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/op.hpp"

namespace cldnn {
using MOE3GemmFusedCompressedOTD = ov::intel_gpu::op::MOE3GemmFusedCompressedOTD;

static size_t get_weights_size(const std::shared_ptr<MOE3GemmFusedCompressedOTD>& op) {
    size_t weights_size = 0;
    for (int i = 0; i < 3; i++)
        weights_size += op->get_weights().gates[i]->get_byte_size();
    for (int i = 0; i < 3; i++)
        weights_size += op->get_weights().ups[i]->get_byte_size();
    for (int i = 0; i < 3; i++)
        weights_size += op->get_weights().downs[i]->get_byte_size();
    return weights_size;
}

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

static void create_weights_memory(cldnn::engine& engine, cldnn::memory::ptr base, cldnn::moe_weights& pw, const std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressedOTD>& op) {
    size_t weights_offset = 0;
    auto weights = op->get_weights();
    auto config = op->get_config();
    auto alloc = [&] (ov::Shape shape, ov::element::Type type) {
        auto format = cldnn::format::get_default_format(shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(type);
        auto layout = cldnn::layout(shape, out_dtype, format);
        auto mem = engine.create_subbuffer(*base, layout, weights_offset);
        weights_offset += layout.bytes_count();
        return mem;
    };
    const size_t group_num = config.hidden_size / config.group_size;
    const size_t group_num2 = config.inter_size / config.group_size;

    pw.gate_w = alloc({config.num_expert * config.inter_size * group_num * config.group_size}, weights.weight_type);
    pw.gate_s = alloc({config.num_expert * config.inter_size * group_num * 1}, weights.scale_type);
    pw.gate_z = alloc({config.num_expert * config.inter_size * group_num * 1}, weights.zp_type);
    pw.up_w = alloc({config.num_expert * config.inter_size * group_num * config.group_size}, weights.weight_type);
    pw.up_s = alloc({config.num_expert * config.inter_size * group_num * 1}, weights.scale_type);
    pw.up_z = alloc({config.num_expert * config.inter_size * group_num * 1}, weights.zp_type);
    pw.down_w = alloc({config.num_expert * config.hidden_size * group_num2 * config.group_size}, weights.weight_type);
    pw.down_s = alloc({config.num_expert * config.hidden_size * group_num2 * 1}, weights.scale_type);
    pw.down_z = alloc({config.num_expert * config.hidden_size * group_num2 * 1}, weights.zp_type);
}

/// @brief moe compressed primitive
/// @details Performs moe compressed
struct moe_3gemm_fused_compressed_otd : public primitive_base<moe_3gemm_fused_compressed_otd> {
    CLDNN_DECLARE_PRIMITIVE(moe_3gemm_fused_compressed_otd)

    moe_3gemm_fused_compressed_otd() : primitive_base("", {}) {}

    // @brief Constructs moe primitive / layer.
    //
    // @param id      An identifier of new primitive.
    // @param inputs  A list of Input primitive ids (inputs).
    //                   0: hidden_states - input tensor with hidden representations
    //                   1: routing_weights - [num_seq, num_experts] routing weights for all experts
    //
    moe_3gemm_fused_compressed_otd(const primitive_id& id, const std::vector<input_info>& inputs, 
                                  const std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressedOTD>& op,
                                  cldnn::memory::ptr base,
                                  moe_weights weights)
        : primitive_base(id, inputs, 1, {optional_data_type()}), 
          m_op(op), m_base(base), m_weights(weights) {}


    // void save(BinaryOutputBuffer& ob) const override {
    //     primitive_base<moe_3gemm_fused_compressed_otd>::save(ob);
    //     ob << make_data(&m_op->get_config(), sizeof(m_op->get_config()));
    //     ob << m_base;
    // }

    // void load(BinaryInputBuffer& ib) override {
    //     primitive_base<moe_3gemm_fused_compressed_otd>::load(ib);
    //     ib >> make_data(&m_op->get_config(), sizeof(m_op->get_config()));
    //     ib >> m_base;
    //     create_weights_memory(ib.get_engine(), m_base, m_weights, m_op);
    // }

    std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressedOTD> m_op;
    cldnn::memory::ptr m_base;
    moe_weights m_weights;
};

}  // namespace cldnn