// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_gathered_moe_to_moe_compressed.hpp"

#include <cmath>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/moe_compressed.hpp"

namespace ov::intel_gpu {

using namespace ov::op;

// Trace back through Fake-Quantization decompression chain:
//   Const(u8) → Convert(f16) → Subtract(zp) → Multiply(scale) → Convert(f32)
// Returns true if the chain matches and fills raw_const, scale_const, zp_const.
static bool trace_fq_chain(const ov::Output<ov::Node>& dequant_output,
                           std::shared_ptr<v0::Constant>& raw_const,
                           std::shared_ptr<v0::Constant>& scale_const,
                           std::shared_ptr<v0::Constant>& zp_const) {
    // dequant_output should be Convert(f32)
    auto final_convert = ov::as_type_ptr<v0::Convert>(dequant_output.get_node_shared_ptr());
    if (!final_convert)
        return false;

    // Input of final Convert should be Multiply(scale)
    auto mul = ov::as_type_ptr<v1::Multiply>(final_convert->input_value(0).get_node_shared_ptr());
    if (!mul)
        return false;

    // Multiply has two inputs: Subtract result and scale Const
    // Either input could be the scale; find the Const one
    std::shared_ptr<ov::Node> sub_node;
    for (size_t i = 0; i < 2; i++) {
        auto c = ov::as_type_ptr<v0::Constant>(mul->input_value(i).get_node_shared_ptr());
        if (c) {
            scale_const = c;
            sub_node = mul->input_value(1 - i).get_node_shared_ptr();
            break;
        }
    }
    if (!scale_const || !sub_node)
        return false;

    auto sub = ov::as_type_ptr<v1::Subtract>(sub_node);
    if (!sub)
        return false;

    // Subtract has: Convert(weight) - Convert(zp) or Convert(weight) - zp_const
    // Input 0 should be Convert(f16) from raw weight
    auto weight_convert = ov::as_type_ptr<v0::Convert>(sub->input_value(0).get_node_shared_ptr());
    if (!weight_convert)
        return false;

    raw_const = ov::as_type_ptr<v0::Constant>(weight_convert->input_value(0).get_node_shared_ptr());
    if (!raw_const)
        return false;

    // Input 1: either direct Const or Convert(Const)
    auto zp_node = sub->input_value(1).get_node_shared_ptr();
    zp_const = ov::as_type_ptr<v0::Constant>(zp_node);
    if (!zp_const) {
        auto zp_convert = ov::as_type_ptr<v0::Convert>(zp_node);
        if (zp_convert) {
            zp_const = ov::as_type_ptr<v0::Constant>(zp_convert->input_value(0).get_node_shared_ptr());
        }
    }
    return zp_const != nullptr;
}

// Check if a node is a Gather with batch_dims=0 and scalar axis const
static std::shared_ptr<v8::Gather> as_gather(const std::shared_ptr<ov::Node>& node) {
    return ov::as_type_ptr<v8::Gather>(node);
}

ConvertGatheredMoeToMoeCompressed::ConvertGatheredMoeToMoeCompressed(bool has_batch_dim) {
    using namespace ov::pass::pattern;

    // Match the end of the MoE block: ReduceSum
    auto reduce_sum_m = wrap_type<v1::ReduceSum>({any_input(), any_input()});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto reduce_sum = ov::as_type_ptr<v1::ReduceSum>(m.get_match_root());
        if (!reduce_sum || !reduce_sum->get_keep_dims() == true) {
            // gpt-oss uses keep_dims=false
        }

        // ReduceSum input 0: Reshape
        auto reshape3 = ov::as_type_ptr<v1::Reshape>(reduce_sum->input_value(0).get_node_shared_ptr());
        if (!reshape3)
            return false;

        // Reshape input 0: Multiply (down_output × routing_weight)
        auto mul_route = ov::as_type_ptr<v1::Multiply>(reshape3->input_value(0).get_node_shared_ptr());
        if (!mul_route)
            return false;

        // The Multiply has two inputs: down_output and routing_weight
        // Routing weight comes from: SoftMax → Reshape
        // Down output comes from: Add(bias_down)
        // We need to figure out which input is which
        std::shared_ptr<ov::Node> down_add_node, routing_reshape_node;
        for (size_t i = 0; i < 2; i++) {
            auto candidate = mul_route->input_value(i).get_node_shared_ptr();
            auto reshape_candidate = ov::as_type_ptr<v1::Reshape>(candidate);
            if (reshape_candidate) {
                // Check if this reshape's input comes from SoftMax
                auto softmax = ov::as_type_ptr<v8::Softmax>(reshape_candidate->input_value(0).get_node_shared_ptr());
                if (softmax) {
                    routing_reshape_node = reshape_candidate;
                    down_add_node = mul_route->input_value(1 - i).get_node_shared_ptr();
                    break;
                }
            }
        }
        if (!routing_reshape_node || !down_add_node)
            return false;

        // down_add_node should be Add(down_squeeze_output, gathered_down_bias)
        auto down_add = ov::as_type_ptr<v1::Add>(down_add_node);
        if (!down_add)
            return false;

        // Follow the routing path: Reshape ← SoftMax ← TopK values
        auto softmax_node = ov::as_type_ptr<v8::Softmax>(
            ov::as_type_ptr<v1::Reshape>(routing_reshape_node)->input_value(0).get_node_shared_ptr());
        if (!softmax_node)
            return false;

        // SoftMax input: TopK output 0 (values)
        auto topk_values_producer = softmax_node->input_value(0).get_node_shared_ptr();
        // The TopK op has 2 outputs: values (port 0) and indices (port 1)
        // Find the TopK node from the values output
        std::shared_ptr<v11::TopK> topk_node;
        if (softmax_node->input_value(0).get_index() == 0) {
            topk_node = ov::as_type_ptr<v11::TopK>(topk_values_producer);
        }
        if (!topk_node)
            return false;

        // TopK input: Add(router_matmul, router_bias) or direct MatMul
        auto topk_input_node = topk_node->input_value(0).get_node_shared_ptr();
        std::shared_ptr<ov::Node> router_matmul_node;
        auto router_add = ov::as_type_ptr<v1::Add>(topk_input_node);
        if (router_add) {
            // Try both inputs for the MatMul
            for (size_t i = 0; i < 2; i++) {
                auto mm = ov::as_type_ptr<v0::MatMul>(router_add->input_value(i).get_node_shared_ptr());
                if (mm) {
                    router_matmul_node = mm;
                    break;
                }
            }
        } else {
            router_matmul_node = ov::as_type_ptr<v0::MatMul>(topk_input_node);
        }
        if (!router_matmul_node)
            return false;

        // Get TopK indices (port 1) — these are used as expert selection indices
        auto topk_indices_output = topk_node->output(1);

        // Find the flattened indices Reshape (TopK indices → Reshape to [-1])
        // The TopK indices may be consumed by Convert then Reshape, or directly by Reshape
        std::shared_ptr<v1::Reshape> indices_reshape;
        for (const auto& user : topk_indices_output.get_target_inputs()) {
            auto node = user.get_node()->shared_from_this();
            auto r = ov::as_type_ptr<v1::Reshape>(node);
            if (r) {
                indices_reshape = r;
                break;
            }
            // Could be Convert → Reshape
            auto convert = ov::as_type_ptr<v0::Convert>(node);
            if (convert) {
                for (const auto& user2 : convert->output(0).get_target_inputs()) {
                    r = ov::as_type_ptr<v1::Reshape>(user2.get_node()->shared_from_this());
                    if (r) {
                        indices_reshape = r;
                        break;
                    }
                }
            }
            if (indices_reshape)
                break;
        }
        if (!indices_reshape)
            return false;

        // The flattened indices are used by Gather ops for weight/bias selection
        auto flat_indices_output = indices_reshape->output(0);

        // ── Trace the down projection path ──────────────────────────────
        // down_add: Add(down_squeeze, gathered_down_bias)
        // Find Squeeze input
        std::shared_ptr<v0::Squeeze> down_squeeze;
        std::shared_ptr<v8::Gather> down_bias_gather;
        for (size_t i = 0; i < 2; i++) {
            auto sq = ov::as_type_ptr<v0::Squeeze>(down_add->input_value(i).get_node_shared_ptr());
            if (sq) {
                down_squeeze = sq;
                auto g = as_gather(down_add->input_value(1 - i).get_node_shared_ptr());
                if (g) down_bias_gather = g;
                break;
            }
        }
        if (!down_squeeze || !down_bias_gather)
            return false;

        // down MatMul: Squeeze ← MatMul(unsqueeze_silu_output, gathered_down_weight)
        auto down_matmul = ov::as_type_ptr<v0::MatMul>(down_squeeze->input_value(0).get_node_shared_ptr());
        if (!down_matmul)
            return false;

        // down_matmul input 1: Gather(dequanted_down_weight, indices)
        auto down_weight_gather = as_gather(down_matmul->input_value(1).get_node_shared_ptr());
        if (!down_weight_gather)
            return false;

        // Trace down weight FQ chain
        auto down_dequant_output = down_weight_gather->input_value(0);
        std::shared_ptr<v0::Constant> down_w_raw, down_scale, down_zp;
        if (!trace_fq_chain(down_dequant_output, down_w_raw, down_scale, down_zp))
            return false;

        // Trace down bias: Gather(dequanted_bias, indices)
        auto down_bias_dequant = down_bias_gather->input_value(0);

        // ── Trace the SwiGLU activation path ────────────────────────────
        // down_matmul input 0: Unsqueeze(multiply_swiglu_output)
        auto down_unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(down_matmul->input_value(0).get_node_shared_ptr());
        if (!down_unsqueeze)
            return false;

        auto swiglu_multiply = ov::as_type_ptr<v1::Multiply>(down_unsqueeze->input_value(0).get_node_shared_ptr());
        if (!swiglu_multiply)
            return false;

        // SwiGLU: Add(Clamp_output, const) * Swish(Minimum_output, beta)
        // Find the Swish and Add branches
        std::shared_ptr<v4::Swish> swish_node;
        std::shared_ptr<v1::Add> clamp_add_node;
        for (size_t i = 0; i < 2; i++) {
            auto sw = ov::as_type_ptr<v4::Swish>(swiglu_multiply->input_value(i).get_node_shared_ptr());
            if (sw) {
                swish_node = sw;
                clamp_add_node = ov::as_type_ptr<v1::Add>(swiglu_multiply->input_value(1 - i).get_node_shared_ptr());
                break;
            }
        }
        if (!swish_node || !clamp_add_node)
            return false;

        // Extract expert_beta from Swish
        float expert_beta = 1.0f;
        if (swish_node->get_input_size() > 1) {
            auto beta_const = ov::as_type_ptr<v0::Constant>(swish_node->input_value(1).get_node_shared_ptr());
            if (beta_const) {
                expert_beta = beta_const->cast_vector<float>()[0];
            }
        }

        // Swish ← Minimum(slice2_output, const)
        auto minimum_node = ov::as_type_ptr<v1::Minimum>(swish_node->input_value(0).get_node_shared_ptr());
        if (!minimum_node)
            return false;

        // Find Slice for the swish branch (slice2)
        std::shared_ptr<v8::Slice> slice2;
        for (size_t i = 0; i < 2; i++) {
            auto s = ov::as_type_ptr<v8::Slice>(minimum_node->input_value(i).get_node_shared_ptr());
            if (s) {
                slice2 = s;
                break;
            }
        }
        if (!slice2)
            return false;

        // Clamp ← ... ← Slice(gate_up_add_output)
        // clamp_add_node: Add(Clamp_output, const)
        std::shared_ptr<v0::Clamp> clamp_node;
        for (size_t i = 0; i < 2; i++) {
            auto c = ov::as_type_ptr<v0::Clamp>(clamp_add_node->input_value(i).get_node_shared_ptr());
            if (c) {
                clamp_node = c;
                break;
            }
        }
        if (!clamp_node)
            return false;

        float expert_alpha = static_cast<float>(clamp_node->get_max());

        // Find the Slice for the clamp branch (slice1)
        auto slice1 = ov::as_type_ptr<v8::Slice>(clamp_node->input_value(0).get_node_shared_ptr());
        if (!slice1)
            return false;

        // Both slices should come from the same Add (gate_up bias add)
        auto gate_up_add_from_slice1 = slice1->input_value(0).get_node_shared_ptr();
        auto gate_up_add_from_slice2 = slice2->input_value(0).get_node_shared_ptr();
        if (gate_up_add_from_slice1 != gate_up_add_from_slice2)
            return false;

        auto gate_up_add = ov::as_type_ptr<v1::Add>(gate_up_add_from_slice1);
        if (!gate_up_add)
            return false;

        // Extract gate_idx from slice2 (swish lane start + step=2)
        size_t gate_idx = 0;
        {
            auto start_c = ov::as_type_ptr<v0::Constant>(slice2->input_value(1).get_node_shared_ptr());
            auto step_c = ov::as_type_ptr<v0::Constant>(slice2->input_value(3).get_node_shared_ptr());
            if (start_c && step_c) {
                auto starts = start_c->cast_vector<int64_t>();
                auto steps = step_c->cast_vector<int64_t>();
                for (size_t i = 0; i < std::min(starts.size(), steps.size()); i++) {
                    if (steps[i] == 2) {
                        gate_idx = static_cast<size_t>(starts[i]);
                        break;
                    }
                }
            }
        }

        // ── Trace the gate_up projection path ───────────────────────────
        // gate_up_add: Add(gate_up_squeeze, gathered_gate_up_bias)
        std::shared_ptr<v0::Squeeze> gate_up_squeeze;
        std::shared_ptr<v8::Gather> gate_up_bias_gather;
        for (size_t i = 0; i < 2; i++) {
            auto sq = ov::as_type_ptr<v0::Squeeze>(gate_up_add->input_value(i).get_node_shared_ptr());
            if (sq) {
                gate_up_squeeze = sq;
                auto g = as_gather(gate_up_add->input_value(1 - i).get_node_shared_ptr());
                if (g) gate_up_bias_gather = g;
                break;
            }
        }
        if (!gate_up_squeeze || !gate_up_bias_gather)
            return false;

        // gate_up MatMul: Squeeze ← MatMul(unsqueeze_input, gathered_gate_up_weight)
        auto gate_up_matmul = ov::as_type_ptr<v0::MatMul>(gate_up_squeeze->input_value(0).get_node_shared_ptr());
        if (!gate_up_matmul)
            return false;

        // gate_up weight Gather
        auto gate_up_weight_gather = as_gather(gate_up_matmul->input_value(1).get_node_shared_ptr());
        if (!gate_up_weight_gather)
            return false;

        // Trace gate_up weight FQ chain
        auto gate_up_dequant_output = gate_up_weight_gather->input_value(0);
        std::shared_ptr<v0::Constant> gate_up_w_raw, gate_up_scale, gate_up_zp;
        if (!trace_fq_chain(gate_up_dequant_output, gate_up_w_raw, gate_up_scale, gate_up_zp))
            return false;

        // Trace gate_up bias: Gather(dequanted_bias, indices)
        auto gate_up_bias_dequant = gate_up_bias_gather->input_value(0);

        // ── Find the hidden states input ────────────────────────────────
        // gate_up_matmul input 0: Unsqueeze ← Gather(input_hidden, indices)
        auto input_unsqueeze = ov::as_type_ptr<v0::Unsqueeze>(gate_up_matmul->input_value(0).get_node_shared_ptr());
        if (!input_unsqueeze)
            return false;

        auto input_gather = as_gather(input_unsqueeze->input_value(0).get_node_shared_ptr());
        if (!input_gather)
            return false;

        // The hidden_states is the input to the Gather (before expert selection)
        auto hidden_states = input_gather->input_value(0);

        // ── Validate weight shapes ──────────────────────────────────────
        auto gate_up_w_shape = gate_up_w_raw->get_shape();
        auto down_w_shape = down_w_raw->get_shape();
        if (gate_up_w_shape.size() != 3 || down_w_shape.size() != 3)
            return false;

        const size_t num_experts = gate_up_w_shape[0];
        const size_t hidden_size = gate_up_w_shape[1];  // K for gate_up (transpose_b=false: [E, K, N])
        const size_t inter_size = gate_up_w_shape[2];   // N for gate_up (= 2 * actual_inter)

        // Verify down_proj: [experts, inter/2, hidden]
        if (down_w_shape[0] != num_experts)
            return false;

        // Get top_k from TopK node
        auto topk_k_const = ov::as_type_ptr<v0::Constant>(topk_node->input_value(1).get_node_shared_ptr());
        size_t top_k = 0;
        if (topk_k_const) {
            top_k = static_cast<size_t>(topk_k_const->cast_vector<int64_t>()[0]);
        } else {
            // Try from output shape
            auto indices_shape = topk_node->get_output_partial_shape(1);
            if (indices_shape.rank().is_static() && indices_shape[indices_shape.rank().get_length() - 1].is_static()) {
                top_k = static_cast<size_t>(indices_shape[indices_shape.rank().get_length() - 1].get_length());
            }
        }
        if (top_k == 0)
            return false;

        // ── Determine group_size from scale shapes ──────────────────────
        auto gate_up_scale_shape = gate_up_scale->get_shape();
        auto down_scale_shape = down_scale->get_shape();
        const size_t down_K = down_w_shape[1];
        const size_t down_num_groups = (down_scale_shape.size() >= 3) ? down_scale_shape[2] : 1;
        const size_t group_size = (down_num_groups <= 1) ? std::numeric_limits<size_t>::max()
                                                         : (down_K / down_num_groups);

        // ── Build MOECompressed inputs ──────────────────────────────────
        // GEMM2 compressed layout for MOECompressed:
        //   0: hidden_states
        //   1: topk_weights (softmax-normalized routing weights from TopK values output)
        //   2: topk_indices (from TopK indices output)
        //   3: gate_up_w (raw u8)
        //   4: gate_up_scale
        //   5: gate_up_zp (if has_zp)
        //   5/6: bias_up (full dequanted, not gathered)
        //   6/7: down_w (raw u8)
        //   7/8: down_scale
        //   8/9: down_zp (if has_zp)
        //   8/10: bias_down (full dequanted, not gathered)

        bool has_zp = true;  // gpt-oss always has zp

        ov::OutputVector moe_inputs;
        moe_inputs.push_back(hidden_states);

        // Routing: TopK values → SoftMax output as topk_weights
        auto softmax_output = softmax_node->output(0);
        moe_inputs.push_back(softmax_output);

        // TopK indices output (unflatttened, original 2D shape)
        moe_inputs.push_back(topk_indices_output);

        // gate_up params
        moe_inputs.push_back(gate_up_w_raw);
        moe_inputs.push_back(gate_up_scale);
        if (has_zp) {
            moe_inputs.push_back(gate_up_zp);
        }
        moe_inputs.push_back(gate_up_bias_dequant);  // full un-gathered dequanted bias

        // down params
        moe_inputs.push_back(down_w_raw);
        moe_inputs.push_back(down_scale);
        if (has_zp) {
            moe_inputs.push_back(down_zp);
        }
        moe_inputs.push_back(down_bias_dequant);  // full un-gathered dequanted bias

        // ── Create config ───────────────────────────────────────────────
        using MOECompressed = ov::op::internal::MOECompressed;
        MOECompressed::Config config{
            {ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP, expert_alpha, expert_beta, gate_idx},
            hidden_size,
            inter_size,
            num_experts,
            0,   // num_shared_expert
            top_k,
            group_size,
            has_batch_dim,
            has_zp,
            ov::element::dynamic,
        };

        auto moe_node = std::make_shared<MOECompressed>(moe_inputs, config);
        moe_node->set_friendly_name(reduce_sum->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), moe_node);

        // The ReduceSum output feeds into the next layer. But check if there's a Reshape after ReduceSum.
        // In gpt-oss, there's a Reshape after ReduceSum to restore the original shape.
        // We need to find that Reshape and replace its output too.
        auto reduce_sum_users = reduce_sum->output(0).get_target_inputs();
        if (reduce_sum_users.size() == 1) {
            auto next = reduce_sum_users.begin()->get_node()->shared_from_this();
            auto end_reshape = ov::as_type_ptr<v1::Reshape>(next);
            if (end_reshape) {
                ov::replace_node(end_reshape, moe_node);
                return true;
            }
        }

        ov::replace_node(reduce_sum, moe_node);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(reduce_sum_m, "ConvertGatheredMoeToMoeCompressed");
    this->register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
