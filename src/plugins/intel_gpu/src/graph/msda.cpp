// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_object.h"
#include "msda_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(msda);

namespace {
// Overload << operator for vectors
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
};  // namespace

std::string msda_inst::to_string(const msda_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    node_info->dump(primitive_description);

    return primitive_description.str();
}

msda_inst::typed_primitive_inst(network& network, const msda_node& node) : parent(network, node) {}

}  // namespace cldnn