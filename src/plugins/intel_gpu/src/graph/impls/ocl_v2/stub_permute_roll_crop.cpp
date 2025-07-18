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

class PermuteRollCrop : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_permute_roll_crop";
    PermuteRollCrop(bool is_shift) : KernelGenerator(m_name, is_shift ? "_permute_roll_crop" : "_permute_crop") {}

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

struct OPENVINO_CORE_EXPORTS permute_roll_crop_opt : cldnn::custom_kernel {
    std::shared_ptr<stub> m_prim;
    program_node& m_prog_node;

    size_t m_window;
    size_t m_shift;
    permute_roll_crop_opt(std::shared_ptr<stub> prim, program_node& prog) : m_prim(prim), m_prog_node(prog) {
        OPENVINO_ASSERT(prim->m_params.count("WINDOW_SIZE"), "attribute 'WINDOW_SIZE' is expected");
        OPENVINO_ASSERT(prim->m_params.count("SHIFT_SIZE"), "attribute 'SHIFT_SIZE' is expected");
        m_window = std::stoi(prim->m_params["WINDOW_SIZE"]);
        m_shift = std::stoi(prim->m_params["SHIFT_SIZE"]);
    }
    virtual std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>> create_kernels() override {
        return {std::make_shared<Stage>(std::make_shared<PermuteRollCrop>(m_shift > 0))};
    }

    virtual cldnn::event::ptr execute(const std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>>&kernels, const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        // node input: attention_window [B*win_no, window*window, C]
        // node input: query_4d [B, H, W, C]
        // kernel input: H, W, attention_window, output
        auto dep = ins.dependencies()[0];
        auto attention_layout = dep.first->get_impl_params()->get_output_layout(dep.second);
        auto query_4d_layout = ins.dependencies()[1].first->get_impl_params()->get_output_layout(dep.second);
        auto query_4d_ps = query_4d_layout.get<ov::PartialShape>();
        OPENVINO_ASSERT(query_4d_ps.rank() == 4, "rank 4 is expected: [B, H, W, C]");
        auto in_b = attention_layout.get_partial_shape()[0].get_length();

        std::vector<size_t> global = {m_window * m_window, static_cast<size_t>(in_b)};
        std::vector<size_t> local = {1, 1};
        cldnn::kernel_arguments_desc desc;
        cldnn::kernel_arguments_data args;

        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        scalars_desc scalars_desc;
        {
            scalar_desc sdesc;
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = query_4d_ps[1].get_length();
            scalars_desc.push_back(sdesc);
        }
        args.scalars = &scalars_desc;

        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        {
            scalar_desc sdesc;
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = query_4d_ps[2].get_length();
            scalars_desc.push_back(sdesc);
        }
    
        desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.inputs.push_back(ins.input_memory_ptr(0));

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
        auto layout = params.input_layouts[1];

        return layout;
    }

    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const override {
        return {calc_output_layout(node, impl_param)};
    }
};

static std::shared_ptr<custom_kernel> create_permute_roll_crop_kernel(std::shared_ptr<stub> prim, program_node& prog) {
    return std::make_shared<ov::intel_gpu::ocl::permute_roll_crop_opt>(prim, prog);
}

}

namespace cldnn {
DEFINE_REG_CUSTOM_KERNEL(PermuteRollCrop, ov::intel_gpu::ocl::create_permute_roll_crop_kernel);
}