// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"

namespace cldnn {

using MOE3GemmFusedCompressed = ov::intel_gpu::op::MOE3GemmFusedCompressed;
using ProgramBuilder = ov::intel_gpu::ProgramBuilder;
extern std::string file_path;
extern size_t offload_to_disk;
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

[[maybe_unused]] static void create_weights_memory(cldnn::engine& engine, cldnn::memory::ptr base, cldnn::moe_weights& pw, 
    const std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>& op, size_t num_expert = 0, size_t weights_offset = 0) {
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
    const size_t group_num = (config.hidden_size / config.group_size > 0) ? (config.hidden_size / config.group_size) : 1;
    const size_t group_num2 = (config.inter_size / config.group_size > 0) ? (config.inter_size / config.group_size) : 1;
    if (num_expert == 0) {
        num_expert = config.num_expert;
    }
    pw.gate_w = alloc({num_expert, config.inter_size, config.hidden_size}, weights.weight_type);
    pw.up_w = alloc({num_expert, config.inter_size, config.hidden_size}, weights.weight_type);
    pw.down_w = alloc({num_expert, config.hidden_size, config.inter_size}, weights.weight_type);

    pw.gate_s = alloc({num_expert, config.inter_size, group_num}, weights.scale_type);
    pw.gate_z = alloc({num_expert, config.inter_size, group_num}, weights.zp_type);
    pw.up_s = alloc({num_expert, config.inter_size, group_num}, weights.scale_type);
    pw.up_z = alloc({num_expert, config.inter_size,  group_num}, weights.zp_type);
    pw.down_s = alloc({num_expert, config.hidden_size, group_num2}, weights.scale_type);
    pw.down_z = alloc({num_expert, config.hidden_size, group_num2}, weights.zp_type);
}

static size_t get_weights_size(const std::shared_ptr<MOE3GemmFusedCompressed>& op) {
    size_t weights_size = 0;
    for (int i = 0; i < 3; i++)
        weights_size += op->get_weights().gates[i]->get_byte_size();
    for (int i = 0; i < 3; i++)
        weights_size += op->get_weights().ups[i]->get_byte_size();
    for (int i = 0; i < 3; i++)
        weights_size += op->get_weights().downs[i]->get_byte_size();
    return weights_size;
}

[[maybe_unused]] static cldnn::memory::ptr pre_allocate_weights(cldnn::engine& engine, const std::shared_ptr<MOE3GemmFusedCompressed>& op, size_t num_expert = 0) {
    auto size = get_weights_size(op);
    if (num_expert != 0) {
        size = size * num_expert / op->get_config().num_expert;
        // std::cout << "wzx debug pre_allocate_weights num_expert:" << num_expert << ", size:" << size << std::endl;
    }
    auto layout = cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(size)}, ov::element::i8, cldnn::format::bfyx);
    auto alloc_type = engine.get_preferred_memory_allocation_type(false);
    auto mem = engine.allocate_memory(layout, alloc_type, false);
    return mem;
}
 
[[maybe_unused]] static void fill_weights_memory(cldnn::engine& engine, const std::shared_ptr<MOE3GemmFusedCompressed>& op, cldnn::moe_weights& wei_mem) {
    auto& stream = engine.get_service_stream();
    auto fill = [&] (const std::shared_ptr<ov::op::v0::Constant>& op, cldnn::memory_ptr mem) {
        if (!mem)
            return;
        ov::Shape const_shape = op->get_shape();
        auto constFormat = cldnn::format::get_default_format(const_shape.size());
        cldnn::data_types out_dtype = cldnn::element_type_to_data_type(op->get_output_element_type(0));
        auto layout = cldnn::layout(const_shape, out_dtype, constFormat);
        auto data = op->get_data_ptr<uint8_t>();
        mem->copy_from(stream, data, 0, 0, layout.bytes_count(), true);
    };

    fill(op->get_weights().gates[0], wei_mem.gate_w);  
    fill(op->get_weights().ups[0], wei_mem.up_w);                                                 
    fill(op->get_weights().downs[0], wei_mem.down_w);

    fill(op->get_weights().gates[1],  wei_mem.gate_s);                                                 
    fill(op->get_weights().gates[2], wei_mem.gate_z);                                                 
    fill(op->get_weights().ups[1], wei_mem.up_s);                                                 
    fill(op->get_weights().ups[2], wei_mem.up_z);                                                 
    fill(op->get_weights().downs[1], wei_mem.down_s);
    fill(op->get_weights().downs[2], wei_mem.down_z); 
 }



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
        primitive_base<moe_3gemm_fused_compressed>::save(ob);
        ob << make_data(&_config, sizeof(_config));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<moe_3gemm_fused_compressed>::load(ib);
        ib >> make_data(&_config, sizeof(_config));
    }
};

}  // namespace cldnn
