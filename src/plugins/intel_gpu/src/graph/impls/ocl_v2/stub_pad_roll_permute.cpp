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

class PadRollPermute : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_pad_roll_permute";
    PadRollPermute(bool is_shift) : KernelGenerator(m_name, is_shift ? "_pad_roll_permute" : "_pad_permute") {}

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

struct OPENVINO_CORE_EXPORTS pad_roll_permute_opt : cldnn::custom_kernel {
    std::shared_ptr<stub> m_prim;
    program_node& m_prog_node;

    size_t m_window;
    size_t m_shift;
    pad_roll_permute_opt(std::shared_ptr<stub> prim, program_node& prog) : m_prim(prim), m_prog_node(prog) {
        OPENVINO_ASSERT(prim->m_params.count("WINDOW_SIZE"), "attribute 'WINDOW_SIZE' is expected");
        OPENVINO_ASSERT(prim->m_params.count("SHIFT_SIZE"), "attribute 'SHIFT_SIZE' is expected");
        m_window = std::stoi(prim->m_params["WINDOW_SIZE"]);
        m_shift = std::stoi(prim->m_params["SHIFT_SIZE"]);
    }
    virtual std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>> create_kernels() override {
        return {std::make_shared<Stage>(std::make_shared<PadRollPermute>(m_shift > 0))};
    }

    virtual cldnn::event::ptr execute(const std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>>&kernels, const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        // node input: query[B, H, W, C]
        // kernel input for WSAttention: seq_len, qkv, mask, output
        // kernel input for SWSAttention: seq_len, qkv, mask, output, rotate_mask, hw_pad
        auto dep = ins.dependencies()[0];
        auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
        auto ps = layout.get<ov::PartialShape>();
        OPENVINO_ASSERT(ps.rank() == 4, "rank 4 is expected: [B, H, W, C]");
        auto out_b = ps[0].get_length() * ((ps[1].get_length() + m_window - 1) / m_window) * ((ps[2].get_length() + m_window - 1)  / m_window);

        std::vector<size_t> global = {m_window * m_window, static_cast<size_t>(out_b)};
        std::vector<size_t> local = {1, 1};
        cldnn::kernel_arguments_desc desc;
        cldnn::kernel_arguments_data args;

        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        scalars_desc scalars_desc;
        {
            scalar_desc sdesc;
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = ps[1].get_length();
            scalars_desc.push_back(sdesc);
        }
        args.scalars = &scalars_desc;

        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        {
            scalar_desc sdesc;
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = ps[2].get_length();
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
        auto layout = params.input_layouts[0];
        auto ps = layout.get<ov::PartialShape>();
        OPENVINO_ASSERT(ps.rank() == 4, "rank 4 is expected: [B, H, W, C]");
        PartialShape out_ps = {-1, static_cast<int>(m_window * m_window), ps[3]};
        if (ps[0].is_static()) {
            out_ps[0] = ps[0].get_length() * ((ps[1].get_length() + m_window - 1) / m_window) * ((ps[2].get_length() + m_window - 1) / m_window);
        }
        layout.set_partial_shape(out_ps);
        return layout;
    }

    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const override {
        return {calc_output_layout(node, impl_param)};
    }
};

static std::shared_ptr<custom_kernel> create_pad_roll_permute_kernel(std::shared_ptr<stub> prim, program_node& prog) {
    return std::make_shared<ov::intel_gpu::ocl::pad_roll_permute_opt>(prim, prog);
}

}

namespace cldnn {
DEFINE_REG_CUSTOM_KERNEL(PadRollPermute, ov::intel_gpu::ocl::create_pad_roll_permute_kernel);
}