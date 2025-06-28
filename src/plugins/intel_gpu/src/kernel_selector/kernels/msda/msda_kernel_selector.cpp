// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msda_kernel_selector.h"
#include "msda_kernel_opt.h"

namespace kernel_selector {
msda_kernel_selector::msda_kernel_selector() {
    Attach<MSDAKernelOpt>();
}

KernelsData msda_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::MSDA);
}
}  // namespace kernel_selector
