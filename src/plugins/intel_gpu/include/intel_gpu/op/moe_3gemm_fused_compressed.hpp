  // Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/moe_compressed.hpp"

namespace ov::intel_gpu::op {


/// \brief MOE3GemmFusedCompressed that support compressed and fused MOE for GEMM3_SWIGLU.
class MOE3GemmFusedCompressed : public MOECompressed {
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
                      std::shared_ptr<ov::op::v0::Constant> down_z
                     ) {
            gates[0] = gate_w;
            gates[1] = gate_s;
            gates[2] = gate_z;
            ups[0] = up_w;
            ups[1] = up_s;
            ups[2] = up_z;
            downs[0] = down_w;
            downs[1] = down_s;
            downs[2] = down_z;
            weight_type = gate_w->get_element_type();
            scale_type = gate_s->get_element_type();
            zp_type = gate_z->get_element_type();
        };
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> gates;
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> ups;
        std::array<std::shared_ptr<ov::op::v0::Constant>, 3> downs;
        ov::element::Type weight_type{};
        ov::element::Type scale_type{};
        ov::element::Type zp_type{};
    };
public:
    OPENVINO_OP("MOE3GemmFusedCompressed", "gpu_opset", MOECompressed);

    MOE3GemmFusedCompressed() = default;

    /// \brief Constructs a MOE3GemmFusedCompressed operation with config only
    /// \param args The input tensors, in the following order:
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_seq, num_experts] routing weights for all experts

    MOE3GemmFusedCompressed(const OutputVector& args, const MOECompressed::Config config) : MOECompressed(args, config) {
        constructor_validate_and_infer_types();
    }
    void initWeights(std::shared_ptr<ov::op::v0::Constant> gate_w,
                                std::shared_ptr<ov::op::v0::Constant> gate_s,
                                std::shared_ptr<ov::op::v0::Constant> gate_z,
                                std::shared_ptr<ov::op::v0::Constant> up_w,
                                std::shared_ptr<ov::op::v0::Constant> up_s,
                                std::shared_ptr<ov::op::v0::Constant> up_z,
                                std::shared_ptr<ov::op::v0::Constant> down_w,
                                std::shared_ptr<ov::op::v0::Constant> down_s,
                                std::shared_ptr<ov::op::v0::Constant> down_z) {
        m_weights = ExpertWeights(gate_w, gate_s, gate_z,
                                  up_w, up_s, up_z,
                                  down_w, down_s, down_z);
    }
    const ExpertWeights& get_weights() const { return m_weights; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
private:
    ExpertWeights m_weights;
};

}  // namespace ov::intel_gpu::op
