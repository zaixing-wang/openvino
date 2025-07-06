// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/stub.hpp"
#include "openvino/core/except.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ov::intel_gpu::ocl {
class Stage;
}

namespace cldnn {

struct custom_kernel {
    virtual std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>> make_stages() = 0;
    virtual cldnn::event::ptr execute(const std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>>&kernels, const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) = 0;
    virtual layout calc_output_layout(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const = 0;
};

struct custom_kernel_info {
    std::function<std::shared_ptr<custom_kernel>(std::shared_ptr<stub> prim, program_node& prog)> make_custom_kernel;
};

void register_custom_kernel(const std::string& type, const custom_kernel_info& info);

#define DEFINE_REG_CUSTOM_KERNEL(type, create)                                 \
    void __reg__##type##__();                                                  \
    void __reg__##type##__() { register_custom_kernel(#type, {create});}
#define CALL_REG_CUSTOM_KERNEL(type) {                                         \
    void __reg__##type##__();                                                  \
    __reg__##type##__();                                                       \
}

template <>
struct typed_program_node<stub> : public typed_program_node_base<stub> {
private:
    using parent = typed_program_node_base<stub>;

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<stub> prim, program& prog);

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);

        return params;
    }
    std::shared_ptr<custom_kernel> m_kernel;
};

using stub_node = typed_program_node<stub>;

template <>
class typed_primitive_inst<stub> : public typed_primitive_inst_base<stub> {
    using parent = typed_primitive_inst_base<stub>;
    using parent::parent;
    using primitive_inst::update_output_memory;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(stub_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(stub_node const& /* node */, kernel_impl_params const& impl_param);
    static std::string to_string(stub_node const& node);
    typed_primitive_inst(network& network, stub_node const& node);
};

using stub_inst = typed_primitive_inst<stub>;
}  // namespace cldnn