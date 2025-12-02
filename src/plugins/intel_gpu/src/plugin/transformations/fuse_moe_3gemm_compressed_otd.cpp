// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_3gemm_compressed_od.hpp"

#include <memory>

#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed_otd.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

FuseMOE3GemmCompressedOTD::FuseMOE3GemmCompressedOTD() {
    auto hidden_state_m = any_input();
    auto routers_m = any_input();
    auto gate_wei_m = wrap_type<ov::op::v0::Constant>();
    auto gate_scale_m = any_input();
    auto gate_zp_m = any_input();
    auto up_wei_m = wrap_type<ov::op::v0::Constant>();
    auto up_scale_m = any_input();
    auto up_zp_m = any_input();
    auto down_wei_m = wrap_type<ov::op::v0::Constant>();
    auto down_scale_m = any_input();
    auto down_zp_m = any_input();

    // moe compressed
    auto moe_compressed_m = wrap_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>({hidden_state_m->output(0),
                                                                            routers_m->output(0),
                                                                         gate_wei_m->output(0),
                                                                         gate_scale_m->output(0),
                                                                         gate_zp_m->output(0),
                                                                         up_wei_m->output(0),
                                                                         up_scale_m->output(0),
                                                                         up_zp_m->output(0),
                                                                         down_wei_m->output(0),
                                                                         down_scale_m->output(0),
                                                                         down_zp_m->output(0)});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe_compressed = ov::as_type_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>(pattern_map.at(moe_compressed_m).get_node_shared_ptr());
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }
        OutputVector args(2);
        args[0] = pattern_map.at(hidden_state_m);
        args[1] = pattern_map.at(routers_m);
        auto gate_w = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(gate_wei_m)).get_node_shared_ptr();
        auto gate_s = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(gate_scale_m)).get_node_shared_ptr();
        auto gate_z = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(gate_zp_m)).get_node_shared_ptr();
        auto up_w = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(up_wei_m)).get_node_shared_ptr();
        auto up_s = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(up_scale_m)).get_node_shared_ptr();
        auto up_z = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(up_zp_m)).get_node_shared_ptr();
        auto down_w = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(down_wei_m)).get_node_shared_ptr();
        auto down_s = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(down_scale_m)).get_node_shared_ptr();
        auto down_z = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(down_zp_m)).get_node_shared_ptr();
        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressedOTD>(args, moe_compressed->get_config(),
                                                                                                        gate_w, gate_s, gate_z,
                                                                                                        up_w, up_s, up_z,
                                                                                                        down_w, down_s, down_z);
        moe_3gemm_fused_compressed->set_friendly_name(moe_compressed->get_friendly_name());
        ov::copy_runtime_info(moe_compressed, moe_3gemm_fused_compressed);
        ov::replace_node(moe_compressed, moe_3gemm_fused_compressed);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_compressed_m, "FuseMOE3GemmCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
