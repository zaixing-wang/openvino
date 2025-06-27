// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi_scale_deformable_attn_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/split.hpp"

using namespace ov::pass::pattern;

namespace ov::intel_gpu {

MultiScaleDeformableAttnFusion::MultiScaleDeformableAttnFusion() {
    using namespace ov::pass::pattern;

    auto input0 = any_input();
    auto crop = wrap_type<ov::op::v1::Split>({input0, any_input()});
    // auto reshape = wrap_type<op::v1::Reshape>({crop, any_input()});
    // auto transpose = wrap_type<op::v1::Transpose>({reshape, any_input()});
    std::cout << "wzx debug hit in" << std::endl;

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        std::cout << "wzx debug hit in in" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();
        // const auto& m_input0 = pattern_map.at(input0).get_node_shared_ptr();

        bool has_crop = pattern_map.count(crop);
        if (!has_crop) {
            std::cout << "wzx debug hit1 " << std::endl;
        } else {
            std::cout << "wzx debug hit2" << std::endl;
        }
        std::cout << "wzx debug hit3" << std::endl;
        // if (it != pattern_map.end()) {
        //     m_fc = it->second.get_node_shared_ptr();
        //     new_fc = std::make_shared<op::FullyConnected>(m_data, m_weights, m_bias, output_type);
        // } else {
        //     m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
        //     if (m_fc->input_values().size() == 4)
        //         new_fc = std::make_shared<op::FullyConnectedCompressed>(m_data,
        //                                                                 m_weights,
        //                                                                 m_bias,
        //                                                                 m_fc->input_value(3),
        //                                                                 output_type);
        //     else
        //         new_fc = std::make_shared<op::FullyConnectedCompressed>(m_data,
        //                                                                 m_weights,
        //                                                                 m_bias,
        //                                                                 m_fc->input_value(3),
        //                                                                 m_fc->input_value(4),
        //                                                                 output_type);
        // }
        // new_fc->set_friendly_name(m_convert->get_friendly_name());
        // copy_runtime_info(m.get_matched_nodes(), new_fc);
        // replace_node(m_convert, new_fc);

        return true;
    };

    std::cout << "wzx debug hit in 1" << std::endl;
    auto m = std::make_shared<ov::pass::pattern::Matcher>(crop, "MultiScaleDeformableAttnFusion");
    std::cout << "wzx debug hit in 2" << std::endl;
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
