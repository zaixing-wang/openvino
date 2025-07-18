// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/runtime/engine.hpp"
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief stub primitive
/// @details Performs stub
struct stub : public primitive_base<stub> {
    CLDNN_DECLARE_PRIMITIVE(stub)

    stub() : primitive_base("", {}) {}

    /// @brief Constructs stub primitive / layer.
    ///
    /// @param id                 An identifier of new primitive.
    /// @param inputs             A list of Input primitive ids (inputs).
    stub(const primitive_id& id,
           const std::vector<input_info>& inputs,
           const std::map<std::string, std::string>& params)
        : primitive_base(id, inputs, 1, {optional_data_type()}),
          m_params(params) {
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const stub>(rhs);

        return m_params == rhs_casted.m_params;
    }
    std::map<std::string, std::string> m_params;
};

}  // namespace cldnn