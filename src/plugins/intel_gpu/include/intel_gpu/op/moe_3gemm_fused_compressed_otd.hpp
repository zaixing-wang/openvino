// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"

namespace ov::intel_gpu::op {

/// \brief MOECompressed experts that support offload experts to disk.
class MOE3GemmFusedCompressedOTD : public MOE3GemmFusedCompressed {
public:
    OPENVINO_OP("MOE3GemmFusedCompressedOTD", "gpu_opset", MOE3GemmFusedCompressed);

    struct ExpertWeights {
        ExpertWeights() = default;
        // 0: weight, 1: scale, 2: zp
        ExpertWeights(std::shared_ptr<ov::op::v0::Constant> gate_w,
                      std::shared_ptr<ov::op::v0::Constant> gate_s,
                      std::shared_ptr<ov::op::v0::Constant> gate_z,
                      std::shared_ptr<ov::op::v0::Constant> up_w,
                      std::shared_ptr<ov::op::v0::Constant> up_s,
                      std::shared_ptr<ov::op::v0::Constant> up_z,
                      std::shared_ptr<ov::op::v0::Constant> down_w,
                      std::shared_ptr<ov::op::v0::Constant> down_s,
                      std::shared_ptr<ov::op::v0::Constant> down_z) {
            gates[0] = gate_w;
            gates[1] = gate_s;
            gates[2] = gate_z;
            ups[0] = up_w;
            ups[1] = up_s;
            ups[2] = up_z;
            downs[0] = down_w;
            downs[1] = down_s;
            downs[2] = down_z;
        };
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> gates;
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> ups;
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> downs;
    };

    MOE3GemmFusedCompressedOTD() = default;
    MOE3GemmFusedCompressedOTD(const OutputVector& args,
                               const MOECompressed::Config config,
                               std::shared_ptr<ov::op::v0::Constant> gate_w,
                               std::shared_ptr<ov::op::v0::Constant> gate_s,
                               std::shared_ptr<ov::op::v0::Constant> gate_z,
                               std::shared_ptr<ov::op::v0::Constant> up_w,
                               std::shared_ptr<ov::op::v0::Constant> up_s,
                               std::shared_ptr<ov::op::v0::Constant> up_z,
                               std::shared_ptr<ov::op::v0::Constant> down_w,
                               std::shared_ptr<ov::op::v0::Constant> down_s,
                               std::shared_ptr<ov::op::v0::Constant> down_z)
        : MOE3GemmFusedCompressed(args, config),
          m_weights(gate_w, gate_s, gate_z, up_w, up_s, up_z, down_w, down_s, down_z){};

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    ExpertWeights m_weights;
};

}  // namespace ov::intel_gpu::op
