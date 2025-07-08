// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/roll_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/glu.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"
#include <climits>
#include <memory>

namespace ov {
namespace pass {

RollFusion::RollFusion() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;
    using namespace gen_pattern;

    auto data = any_input();

    auto slice1 = makePattern<ov::op::v1::StridedSlice>({data, {0, 6}, {0ll, LLONG_MAX}, {1, 1}});
    auto slice2 = makePattern<ov::op::v1::StridedSlice>({data, {0, 0}, {0, 6}, {1, 1}});
    auto concat1 = makePattern<ov::op::v0::Concat>({slice2, slice2}, {{"axis", 1}});
    auto slice3 = makePattern<ov::op::v1::StridedSlice>({concat1, {0, 0, 6}, {0ll, 0ll, LLONG_MAX}, {1, 1, 1}});
    auto slice4 = makePattern<ov::op::v1::StridedSlice>({concat1, {0, 0, 0}, {0, 0, 6}, {1, 1, 1}});
    auto concat2 = makePattern<ov::op::v0::Concat>({slice4, slice4}, {{"axis", 2}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(data));

        auto data_output = pattern_map.at(data);
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 2});
        auto shift = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {6, 6});;
        auto roll = std::make_shared<ov::op::v7::Roll>(data_output,
                                                       shift,
                                                       axis);
        roll->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), roll);
        ov::replace_node(m.get_match_root(), roll);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat2, "RollFusion");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
