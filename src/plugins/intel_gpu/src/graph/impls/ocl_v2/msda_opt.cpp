// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msda_opt.hpp"

#include "msda_inst.h"
#include "common_utils/dispatch_utils.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"
#include "intel_gpu/primitives/msda.hpp"


namespace ov::intel_gpu::ocl {
namespace {

class MSDAOptGenerator : public KernelGenerator {
public:
    MSDAOptGenerator() : KernelGenerator("msda") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const auto& desc = params.typed_desc<msda>();

            auto& wgs = kd.params.workGroups;

            // to update
            wgs.global = {1, 1, 1};
            wgs.local = {1, 1, 1};
        }};
    }
};

class MSDAOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MSDAOptImpl)

    Stage::Ptr msda = make_stage<MSDAOptGenerator>();

    MSDAOptImpl() : PrimitiveImplOCL(MSDAOptImplementationManager::get_type_info_static()) {}
    MSDAOptImpl(const program_node& node, const RuntimeParams& params) : MSDAOptImpl() {
        add_stage(msda, params);
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MSDAOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> MSDAOptImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<msda>());
    return std::make_unique<MSDAOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::msda)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MSDAOptImpl)
