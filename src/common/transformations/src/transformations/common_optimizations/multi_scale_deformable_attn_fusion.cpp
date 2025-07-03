// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/multi_scale_deformable_attn_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/block_collection.hpp"
#include "transformations/utils/utils.hpp"

#include "transformations/utils/block_collection.hpp"

#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/power.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "openvino/opsets/opset12.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;
using namespace ov::pass::pattern;
using namespace ov::opset12;

namespace {

ov::pass::pattern::op::Predicate check_input(std::shared_ptr<Node> expected_input) {
    return ov::pass::pattern::op::Predicate(
        [=](const Output<Node>& output) -> bool {
            auto graph_node = output.get_node_shared_ptr();
            auto pattern_node = expected_input.get();
            for (size_t i = 0; i < graph_node->get_input_size(); i++) {
                auto input_node = graph_node->input_value(i).get_node();
                if (pattern_node == input_node) return true;
            }
            
            return false;
        },
        "check_input");
}

std::shared_ptr<ov::Node> grid_sample_block(const std::shared_ptr<ov::Node>& input_attn_value, const std::shared_ptr<ov::Node>& input_attn_offsets) {
    auto attn_Slice = wrap_type<StridedSlice>({input_attn_value, any_input(), any_input(), any_input()});
    auto attn_Reshape_4 = wrap_type<Reshape>({attn_Slice, any_input()});
    auto attn_Transpose = wrap_type<Transpose>({attn_Reshape_4, any_input()});
    auto attn_Reshape_5 = wrap_type<Reshape>({attn_Transpose, any_input()});
    
    auto attn_Gather_9 = wrap_type<Gather>({input_attn_offsets, any_input(), any_input()});
    auto attn_squeeze = std::make_shared<Squeeze>(attn_Gather_9, any_input());  // FIXME???
    auto attn_Transpose_1 = wrap_type<Transpose>({attn_squeeze, any_input()});
    auto attn_Reshape_6 = wrap_type<Reshape>({attn_Transpose_1, any_input()});

    auto attn_GridSample = wrap_type<GridSample>({attn_Reshape_5, attn_Reshape_6});
    auto attn_Unsqueeze_31 = wrap_type<Reshape>({attn_GridSample, any_input()});

    return std::make_shared<pattern::op::Block>(OutputVector{input_attn_value, input_attn_offsets}, OutputVector{attn_Unsqueeze_31}, "grid_sample_block");
}

}  // namespace

MultiScaleDeformableAttnFusion::MultiScaleDeformableAttnFusion() : MultiMatcher("MultiScaleDeformableAttnFusion") {
    using namespace ov::opset12;

    // Pattern 1
    auto attn_value_input = any_input();
    auto attn_offsets_input = any_input();
    auto grid_sampler_block = grid_sample_block(attn_value_input, attn_offsets_input);

    //({flatten_Slice_1194, {-1}}, {{"axis", 0}});
    // ({Unsqueeze_65524 | Unsqueeze_28998, Unsqueeze_65525 | Unsqueeze_28999},
    // wrap_type<opset1::Concat>(pattern::consumers_count(1));
    auto attn_Concat_17 = wrap_type<Concat>(/*check_input(grid_sampler_block),*/ {{"axis", -2}});
    auto attn_Reshape_17 = wrap_type<Reshape>({attn_Concat_17, any_input()});

    std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

    // Pattern 2
    auto attn_weight_input = any_input();
    auto grid_sample_input = any_input();

    auto attn_Transpose_8 = wrap_type<Transpose>({attn_weight_input, any_input()});
    auto attn_Reshape_16 = wrap_type<Reshape>({attn_Transpose_8, any_input()});
    auto attn_Mul_3 = wrap_type<Multiply>({grid_sample_input, attn_Reshape_16});
    auto attn_ReduceSum = wrap_type<ReduceSum>({attn_Mul_3, any_input()});
    auto attn_Reshape_18 = wrap_type<Reshape>({attn_ReduceSum, any_input()});
    auto attn_output_proj_MatMul_transpose_a = wrap_type<Transpose>({attn_Reshape_18, any_input()});

    auto callback = [=](const std::unordered_map<std::shared_ptr<Node>, std::vector<PatternValueMap>>& matches) {
        std::cout << "wzx debug hit in in" << __LINE__ << ", matches.size()=" << matches.size() << std::endl;
        if (matches.size() != 2) {
            return;
        }

        std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

        std::unordered_set<Node*> post_msda_proj;
        std::unordered_map<Node*, const PatternValueMap*> node_to_output_proj_pm;
        for (const auto& pm : matches.at(attn_output_proj_MatMul_transpose_a)) {
            auto root = pm.at(attn_output_proj_MatMul_transpose_a).get_node();
            post_msda_proj.insert(root);
            node_to_output_proj_pm[root] = &pm;
            std::cout << "wzx debug hit in in" << __LINE__ << ", root=" << root->get_friendly_name() << std::endl;
        }

        std::unordered_map<Node*, const PatternValueMap*> node_to_grid_concat_pm;
        for (const auto& pm : matches.at(attn_Reshape_17)) {
            auto root = pm.at(attn_Reshape_17).get_node_shared_ptr();
            node_to_grid_concat_pm[root.get()] = &pm;
            std::cout << "wzx debug hit in in" << __LINE__ << ", root=" << root->get_friendly_name() << std::endl;
        }

        std::cout << "wzx debug hit in in" << __LINE__ << std::endl;

    };

    register_patterns({attn_Reshape_17, attn_output_proj_MatMul_transpose_a}, callback, true);
}