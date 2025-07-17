// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct MSDAPatternShapeParams {
    ov::PartialShape value_shape;
    ov::PartialShape offset_shape;
    ov::PartialShape weight_shape;
};

typedef std::tuple<MSDAPatternShapeParams>
    MSDAPatternParams;

class MSDAPattern : public testing::WithParamInterface<MSDAPatternShapeParams>,
                    public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MSDAPatternShapeParams> obj);
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
