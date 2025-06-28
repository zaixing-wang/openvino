// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msda_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

JitConstants MSDAKernelBase::GetJitConstants(const msda_params& params) const {
    auto jit = MakeBaseParamsJitConstants(params);
    return jit;
}

bool MSDAKernelBase::Validate(const Params& p) const {
    // if (p.GetType() != KernelType::MSDA) {
    //     return false;
    // }

    // const msda_params& params = static_cast<const msda_params&>(p);

    // for (size_t i = 0; i < params.inputs.size(); i++) {
    //     if (params.inputs[i].Dimentions() != 4)
    //         return false;
    // }

    // if (params.outputs[0].Dimentions() != 4)
    //     return false;

    return true;
}
}  // namespace kernel_selector