// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "stub_opt.hpp"

#include <cctype>
#include <cstdint>
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

class PermuteAddConcat : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_permute_add_concat";
    PermuteAddConcat() : KernelGenerator(m_name, "_cm") {}

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

        jit.make("KERNEL_NAME", get_entry_point(params));

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

struct OPENVINO_CORE_EXPORTS permute_add_concat_opt : cldnn::custom_kernel {
    std::shared_ptr<stub> m_prim;
    program_node& m_prog_node;

    permute_add_concat_opt(std::shared_ptr<stub> prim, program_node& prog) : m_prim(prim), m_prog_node(prog) {
    }
    virtual std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>> create_kernels() override {
        return {std::make_shared<Stage>(std::make_shared<PermuteAddConcat>())};
    }

    virtual cldnn::event::ptr execute(const std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>>&kernels, const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        // node input: pos0-3 [B, L, C]
        // node input: bias0-3 [C]
        // output: [B, H*W, C]
        auto len = (ins.get_impl_params()->input_layouts.size() - 1) / 2;
        auto layout = ins.get_output_layout();
        auto ps = layout.get_partial_shape();
        auto b = ps[0].get_length();
        auto l = ps[1].get_length();

        std::vector<size_t> global = {static_cast<size_t>(l), static_cast<size_t>(b)};
        std::vector<size_t> local = {1, 1};
        cldnn::kernel_arguments_desc desc;
        cldnn::kernel_arguments_data args;

        scalars_desc scalars_desc;
        args.scalars = &scalars_desc;
        for (uint32_t i = 0; i < len; i++) {
            auto&& ps = ins.get_impl_params()->input_layouts[i].get_partial_shape();
            desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, i});
            {
                scalar_desc sdesc;
                sdesc.t = scalar_desc::Types::INT32;
                sdesc.v.s32 = ps[1].get_length();
                scalars_desc.push_back(sdesc);
            }
        }
        for (uint32_t i = 0; i < len * 2; i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(ins.input_memory_ptr(i));
        }
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
        auto len = (params.input_layouts.size() - 1) / 2;
        auto layout = params.input_layouts[0];
        auto ps = layout.get_partial_shape();
        PartialShape out_ps = {ps[0], -1, ps[2]};
        if (ps.rank().is_static() && ps[1].is_static()) {
            size_t tokens = 0;
            for (size_t i = 0; i < len; i++) {
                auto&& input_ps = params.input_layouts[i].get_partial_shape();
                tokens += input_ps[1].get_length();
            }
            out_ps[1] = tokens;
        }
        layout.set_partial_shape(out_ps);

        return layout;
    }

    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const override {
        return {calc_output_layout(node, impl_param)};
    }
};

static std::shared_ptr<custom_kernel> create_permute_add_concat_kernel(std::shared_ptr<stub> prim, program_node& prog) {
    return std::make_shared<ov::intel_gpu::ocl::permute_add_concat_opt>(prim, prog);
}

}

namespace cldnn {
DEFINE_REG_CUSTOM_KERNEL(PermuteAddConcat, ov::intel_gpu::ocl::create_permute_add_concat_kernel);
}