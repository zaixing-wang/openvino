// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "stub_opt.hpp"

#include <cctype>
#include <initializer_list>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

#include "cm/utils/kernel_generator.hpp"
#include "cm/utils/kernels_db.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "debug_helper.hpp"
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
#include "cm/utils/kernel_generator.hpp"
#include "stub_helper.hpp"

namespace ov::intel_gpu::ocl {

class AttnCM : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_swattn_cm";
    AttnCM() : KernelGenerator(m_name, "_sdpa_qkv_fused") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<stub>();
        const auto& param = desc->m_params;
        for (auto&& kv : param) {
            auto&& key = kv.first;
            if (std::all_of(key.begin(), key.end(), [](unsigned char c){ return !std::isalpha(c) || std::isupper(c); })) {
                jit.make(kv.first, kv.second);
            }
        }

        jit.make("CMFLA_IS_CAUSAL", 0);
        jit.make("KERNEL_NAME", get_entry_point(params));

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }

    std::string get_build_options(const RuntimeParams& params) const override {
        // -mdump_asm
        return " -cmc -Qxcm_jit_option=\"-abortonspill\" -Qxcm_register_file_size=256 -g2 ";
    }
};

struct OPENVINO_CORE_EXPORTS attn_opt : cldnn::custom_kernel {
    std::shared_ptr<stub> m_prim;
    program_node& m_prog_node;
    size_t m_num_heads;
    attn_opt(std::shared_ptr<stub> prim, program_node& prog) : m_prim(prim), m_prog_node(prog) {
        OPENVINO_ASSERT(prim->m_params.count("NUM_HEADS"), "attribute 'NUM_HEADS' is expected");
        OPENVINO_ASSERT(prim->m_params.count("NUM_KV_HEADS"), "attribute 'NUM_KV_HEADS' is expected");
        OPENVINO_ASSERT(prim->m_params.count("SCALE"), "attribute 'SCALE' is expected");
        OPENVINO_ASSERT(prim->m_params.count("HEAD_SIZE"), "attribute 'HEAD_SIZE' is expected");
        m_num_heads = std::stoi(prim->m_params["NUM_HEADS"]);
    }
    virtual std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>> make_stages() override {
        return {std::make_shared<Stage>(std::make_shared<AttnCM>())};
    }

    virtual cldnn::event::ptr execute(const std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>>&kernels, const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        auto dep = ins.dependencies()[0];
        auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
        // wg_size = 16
        // q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        // wg_seq_len = wg_size * q_step
        // wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
        // GWS = [1, self.num_heads, wg_count * wg_size]
        // LWS = [1, 1, wg_size]
        auto ps = layout.get_partial_shape();
        size_t batch = ps[0].get_length();
        size_t seq_len = ps[1].get_length();
        size_t wg_size = 16;
        auto CM_GRF_WIDTH = 512;
        auto q_step = CM_GRF_WIDTH / 32;
        auto wg_seq_len = wg_size * q_step;
        auto wg_count = (seq_len + wg_seq_len - 1) / wg_seq_len;
        std::vector<size_t> global = {batch, m_num_heads, static_cast<size_t>(wg_count * wg_size)};
        std::vector<size_t> local = {1, 1, wg_size};
        cldnn::kernel_arguments_desc desc;
        cldnn::kernel_arguments_data args;

        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        scalars_desc scalars_desc;
        scalar_desc sdesc;
        sdesc.t = scalar_desc::Types::INT32;
        sdesc.v.s32 = seq_len;
        scalars_desc.push_back(sdesc);
        args.scalars = &scalars_desc;

        desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.inputs.push_back(ins.input_memory_ptr(0));

        desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.inputs.push_back(ins.input_memory_ptr(1));

        desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.outputs.push_back(ins.output_memory_ptr(0));

        desc.workGroups.global = global;
        desc.workGroups.local = local;

        return execute_stage(events,
                             ins,
                             *kernels[0],
                             desc,
                             args,
                             ins.needs_completion_event());
    }
    
    virtual layout calc_output_layout(const program_node& node, const kernel_impl_params& params) const override {
        auto layout = params.input_layouts[0];
        auto ps = layout.get<ov::PartialShape>();
        if (ps.rank().is_static()) {
            OPENVINO_ASSERT(ps.rank().get_length() == 3);
            ps[2] = ps[2] / 3;
        }
        layout.set_partial_shape(ps);
        return layout;
    }

    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const override {
        auto layout = impl_param.input_layouts[0];
        auto ps = layout.get<ov::PartialShape>();
        if (ps.rank().is_static()) {
            OPENVINO_ASSERT(ps.rank().get_length() == 3);
            ps[2] = ps[2] / 3;
        }
        layout.set_partial_shape(ps);
        return {layout};
    }
};

static std::shared_ptr<custom_kernel> create_attn_kernel(std::shared_ptr<stub> prim, program_node& prog) {
    return std::make_shared<ov::intel_gpu::ocl::attn_opt>(prim, prog);
}

}

namespace cldnn {
DEFINE_REG_CUSTOM_KERNEL(WSAttention, ov::intel_gpu::ocl::create_attn_kernel);
}