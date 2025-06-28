// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msda_kernel_opt.h"
#include "kernel_selector_utils.h"
#include "common_types.h"
#include <string>
#include <vector>

namespace kernel_selector {

namespace {
constexpr size_t subgroup_size = 16;
}  // namespace

ParamsKey MSDAKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

bool MSDAKernelOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        return false;

    return true;
}

JitConstants MSDAKernelOpt::GetJitConstants(const msda_params& params, size_t kernel_idx) const {
    auto jit = MSDAKernelBase::GetJitConstants(params);

    // const auto& config = params.conf;
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size));

    return jit;
}

CommonDispatchData MSDAKernelOpt::SetDefault(const msda_params& params) const {
    CommonDispatchData dispatch_data;

    // const auto& query_input = params.inputs[0];
    // if (!query_input.is_dynamic()) {
    //     if (params.conf.is_paged_attention) {
    //         OPENVINO_ASSERT(kernel_idx == KernelsTypes::MULTI_TOKENS);

    //         const size_t heads_num = static_cast<size_t>(params.conf.heads_num);
    //         const size_t head_size = static_cast<size_t>(params.conf.v_head_size);
    //         const size_t sg_num_scale = get_sg_number_scale_factor(params, head_size, kernel_idx);
    //         const size_t target_seq_len_block_size = get_target_seq_len_block_size();
    //         const size_t target_seq_len = static_cast<size_t>(params.conf.paged_attention_aligned_seq_len);

    //         dispatch_data.gws = { heads_num,
    //                               CeilDiv(target_seq_len, target_seq_len_block_size),
    //                               head_size * sg_num_scale };
    //         dispatch_data.lws = { 1, 1, head_size * sg_num_scale };

    //         return dispatch_data;
    //     }

    //     TransposedDimensionAccessHelperBase dims_q(params.inputs[0], params.input0_order);
    //     TransposedDimensionAccessHelperBase output(params.outputs[0], params.output_order);

    //     const size_t batch_size = output.b_dim().v;
    //     const size_t heads_num = output.f_dim().v;
    //     const size_t target_seq_len = dims_q.y_dim().v;
    //     const size_t head_size = static_cast<size_t>(params.conf.v_head_size);
    //     const size_t num_of_partitions = get_partitions_num(params, kernel_idx);
    //     const size_t target_seq_len_block_size = kernel_idx == 1 ? get_target_seq_len_block_size() : 1;

    //     if (kernel_idx == KernelsTypes::SINGLE_TOKEN) {
    //         const size_t sg_num_scale = get_sg_number_scale_factor(params, head_size, kernel_idx);
    //         dispatch_data.gws = { batch_size * heads_num,
    //                               CeilDiv(target_seq_len, target_seq_len_block_size),
    //                               head_size * num_of_partitions * sg_num_scale };
    //         dispatch_data.lws = { 1, 1, head_size * sg_num_scale };
    //     } else if (kernel_idx == KernelsTypes::MULTI_TOKENS) {
    //         const size_t sg_num_scale = get_sg_number_scale_factor(params, head_size, kernel_idx);
    //         dispatch_data.gws = { batch_size * heads_num,
    //                               CeilDiv(target_seq_len, target_seq_len_block_size),
    //                               Align(head_size * sg_num_scale, subgroup_size) };
    //         dispatch_data.lws = { 1, 1, Align(head_size * sg_num_scale, subgroup_size) };
    //     } else if (kernel_idx == KernelsTypes::FINALIZATION) {
    //         dispatch_data.gws = { batch_size * heads_num,
    //                               target_seq_len,
    //                               head_size };
    //         dispatch_data.lws = { 1, 1, head_size };
    //     }
    // }

    return dispatch_data;
}

KernelsData MSDAKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<msda_params>(params);
    // msda_params& newParams = *static_cast<msda_params*>(kd.params.get());

    // auto dispatchData = SetDefault(newParams);
    // auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    // auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    // auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    // GetUpdateDispatchDataFunc(kd);

    // auto& kernel = kd.kernels[0];
    // FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
    //                  EXE_MODE_DEFAULT, false, false, 1,
    //                  GetFusedPrimitiveInputsCount(params), 1, newParams.is_shape_agnostic);

    // if (!newParams.inputActivationParams.empty()) {
    //     kernel.params.arguments.push_back({ArgumentDescriptor::Types::SLOPE, 0});
    // }

    return {kd};
}

void MSDAKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    // kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
    //     const auto& prim_params = static_cast<const msda_params&>(params);

    //     const size_t paged_attention_kernels_num = 1;
    //     const size_t expected_kernels_num =
    //         prim_params.conf.is_paged_attention || unaligned_head_size(prim_params) ? paged_attention_kernels_num : KernelsTypes::TOTAL_KERNELS_NUM;
    //     OPENVINO_ASSERT(kernel_data.kernels.size() == expected_kernels_num, "[GPU] Invalid kernels size for update dispatch data func of MSDA kernel");

    //     if (prim_params.conf.is_paged_attention || unaligned_head_size(prim_params)) {
    //         auto dispatch_data = SetDefault(prim_params, KernelsTypes::MULTI_TOKENS);
    //         kernel_data.kernels[0].params.workGroups.global = dispatch_data.gws;
    //         kernel_data.kernels[0].params.workGroups.local = dispatch_data.lws;
    //         kernel_data.kernels[0].skip_execution = false;

    //         if (prim_params.outputs.size() > 1) {
    //             const auto max_seq_len = prim_params.conf.paged_attention_max_len;
    //             const auto seq_len_partition_size = get_seq_len_partition_size(params, prim_params.conf.v_head_size, KernelsTypes::MULTI_TOKENS);

    //             kernel_data.kernels[0].params.scalars.resize(1);
    //             kernel_data.kernels[0].params.scalars[0].t = ScalarDescriptor::Types::UINT32;
    //             kernel_data.kernels[0].params.scalars[0].v.u32 = static_cast<uint32_t>(Align(max_seq_len, seq_len_partition_size));
    //         }
    //     } else {
    //         const auto num_of_partitions = get_partitions_num(prim_params, KernelsTypes::SINGLE_TOKEN);
    //         const auto buf_sizes = get_internal_buffer_sizes(prim_params, KernelsTypes::SINGLE_TOKEN);
    //         const auto is_prefill = is_prefill_stage(prim_params);

    //         ScalarDescriptor num_of_partitions_scalar;
    //         num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
    //         num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);

    //         auto dispatch_data1 = SetDefault(prim_params, KernelsTypes::SINGLE_TOKEN);
    //         kernel_data.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.global = dispatch_data1.gws;
    //         kernel_data.kernels[KernelsTypes::SINGLE_TOKEN].params.workGroups.local = dispatch_data1.lws;
    //         kernel_data.kernels[KernelsTypes::SINGLE_TOKEN].skip_execution = is_prefill;

    //         auto dispatch_data2 = SetDefault(prim_params, KernelsTypes::MULTI_TOKENS);
    //         kernel_data.kernels[KernelsTypes::MULTI_TOKENS].params.workGroups.global = dispatch_data2.gws;
    //         kernel_data.kernels[KernelsTypes::MULTI_TOKENS].params.workGroups.local = dispatch_data2.lws;
    //         kernel_data.kernels[KernelsTypes::MULTI_TOKENS].skip_execution = !is_prefill;

    //         auto dispatch_data3 = SetDefault(prim_params, KernelsTypes::FINALIZATION);
    //         kernel_data.kernels[KernelsTypes::FINALIZATION].params.workGroups.global = dispatch_data3.gws;
    //         kernel_data.kernels[KernelsTypes::FINALIZATION].params.workGroups.local = dispatch_data3.lws;
    //         kernel_data.kernels[KernelsTypes::FINALIZATION].skip_execution = is_prefill || num_of_partitions == 1;

    //         kernel_data.kernels[KernelsTypes::FINALIZATION].params.scalars.clear();
    //         kernel_data.kernels[KernelsTypes::FINALIZATION].params.scalars.push_back(num_of_partitions_scalar);

    //         kernel_data.internalBuffers.clear();
    //         kernel_data.internalBuffers.push_back(buf_sizes[0]);
    //         kernel_data.internalBuffers.push_back(buf_sizes[0]);
    //         kernel_data.internalBuffers.push_back(buf_sizes[1]);
    //         kernel_data.internalBufferDataType = prim_params.inputs[0].GetDType();
    //     }
    // };
}

KernelsPriority MSDAKernelOpt::GetKernelsPriority(const Params& params) const {
    return params.engineInfo.supports_immad ?  FORCE_PRIORITY_2 : FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
