// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stub_opt.hpp"

#include <initializer_list>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

#include "common_utils/jitter.hpp"
#include "debug_helper.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/stub.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "stub_inst.h"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class StubOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::StubOptImpl)
    std::vector<std::shared_ptr<Stage>> m_stages;
    std::shared_ptr<custom_kernel> m_custom_kernel;

    StubOptImpl() : PrimitiveImplOCL(StubOpt::get_type_info_static()) {}
    virtual ~StubOptImpl() {
    }
    StubOptImpl(const program_node& node, const RuntimeParams& params) : StubOptImpl() {
        const auto& stubop = node.as<stub>();
        m_custom_kernel = stubop.m_kernel;
        m_stages = m_custom_kernel->make_stages();

        for (size_t i = 0; i < m_stages.size(); i++) {
            _stages.push_back(m_stages[i].get());
            _order.push_back(i);
            m_stages[i]->kd = m_stages[i]->codegen->get_kernel_data(params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<StubOptImpl>(this);
        copy->m_stages = m_custom_kernel->make_stages();
        copy->_stages.resize(_stages.size(), 0);
        for (size_t i = 0; i < _stages.size(); i++) {
            copy->_stages[i] = copy->m_stages[i].get();
            copy->_stages[i]->kd = _stages[i]->kd;
            if (_stages[i]->kernel) {
                copy->_stages[i]->kernel = _stages[i]->kernel->clone();
            }
        }
        copy->m_custom_kernel = m_custom_kernel;
        return copy;
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        auto cur_stubop = params.typed_desc<stub>();
        std::vector<BufferDescriptor> internal_buffers;
        // const auto& config = cur_stubop->_config;
        // int max_topk = static_cast<int>(config.topk);
        // int expert_num = static_cast<int>(config.expert_num);

        // auto hidden_states_layout = params.input_layouts[0];
        // auto batch = static_cast<int>(hidden_states_layout.get_shape()[0]);
        // auto data_type = hidden_states_layout.data_type;

        // // softmax+topk
        // layout layout_topk_id(ov::PartialShape{batch, max_topk}, data_types::u32, cldnn::format::bfyx);
        // layout layout_topk_weights(ov::PartialShape{batch, max_topk}, data_type, cldnn::format::bfyx);
        // internal_buffers.emplace_back(layout_topk_id, true);       // topk_id
        // internal_buffers.emplace_back(layout_topk_weights, true);  // topk_weights
        // // fast single batch: scratch.up = up(x) * silu(gate(x)); scratch.y = down(scratch.up) * weight[expert_no]
        // layout layout_gateup_out(ov::PartialShape{batch, static_cast<int>(config.intermediate_size)}, data_type, cldnn::format::bfyx);
        // layout layout_down_out(ov::PartialShape{batch, static_cast<int>(config.hidden_size)}, data_type, cldnn::format::bfyx);
        // internal_buffers.emplace_back(layout_gateup_out, true);  // up
        // internal_buffers.emplace_back(layout_down_out, true);    // y
        // // onednn: scratch.x, scratch.routing_weights = gather(x, ...)
        // //         scratch.up = up(scratch.x)
        // //         scratch.gate = gate(scratch.x) * scratch.up
        // //         scratch.y = down(scratch.gate) * routing_weights
        // internal_buffers.emplace_back(layout_down_out, true);  // x, scratch.x has same layout with down output
        // layout routing_layout(ov::PartialShape{batch * max_topk}, data_type, cldnn::format::bfyx);
        // internal_buffers.emplace_back(layout_down_out, true);    // routing_weights
        // internal_buffers.emplace_back(layout_gateup_out, true);  // gate, scratch.gate has same layout with up
        // // expert masks for gpu
        // layout index_layout(ov::PartialShape{batch}, ov::element::i32, cldnn::format::bfyx);
        // for (int i = 0; i < expert_num; i++) {
        //     internal_buffers.emplace_back(index_layout, true);  // batch
        //     internal_buffers.emplace_back(index_layout, true);  // topk
        // }

        return internal_buffers;
    }

    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance,
                                    Stage& stage,
                                    std::vector<memory::ptr> inputs,
                                    std::vector<memory::ptr> outputs,
                                    const std::vector<size_t>& global,
                                    const std::vector<size_t>& local,
                                    bool needs_completion_event = false) const {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("StubOPOptImpl::execute_stage"));
        cldnn::stream& stream = instance.get_network().get_stream();
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc desc;
        for (uint32_t i = 0; i < inputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
        }

        for (uint32_t i = 0; i < outputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            args.outputs.push_back(outputs[i]);
        }

        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;

        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, needs_completion_event);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("StubOPOptImpl::execute"));
        auto& instance = reinterpret_cast<typed_primitive_inst<stub>&>(ins);
        return instance.get_node().as<stub>().m_kernel->execute(m_stages, events, ins);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> StubOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<stub>());
    return std::make_unique<StubOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::stub)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::StubOptImpl)
