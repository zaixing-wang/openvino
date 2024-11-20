// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// tmp_buf       [batch, heads_num, q_len, kv_len]

// When handling long sequences and executing in FP16, accuracy can significantly vary based on two factors:
// 1) The order of scale application (which can be controlled using the APPLY_SCALE_TO_QUERY macro)
// 2) The type of SoftMax accumulator

inline uint FUNC(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, y, x);
#else
#if INPUT0_DIMS == 4
    return INPUT0_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error sdpa_ref.cl : Unsupported input 0 format
#endif
#endif
}

inline uint FUNC(get_input0_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT0_DIMS_ORDER
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT0_DIMS_ORDER);
#else
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

inline uint FUNC(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef DO_BROADCAST_KEY_VALUE
    DO_BROADCAST_KEY_VALUE;
#endif
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, y, x);
#else
#if INPUT1_DIMS == 4
    return INPUT1_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT1_DIMS == 5
    return INPUT1_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT1_DIMS == 6
    return INPUT1_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error sdpa_ref.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input1_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT1_DIMS_ORDER
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
#else
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

inline uint FUNC(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef DO_BROADCAST_KEY_VALUE
    DO_BROADCAST_KEY_VALUE;
#endif
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, y, x);
#else
#if INPUT2_DIMS == 4
    return INPUT2_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT2_DIMS == 5
    return INPUT2_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT2_DIMS == 6
    return INPUT2_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error sdpa_ref.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input2_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INPUT2_DIMS_ORDER
    return FUNC_CALL(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT2_DIMS_ORDER);
#else
    return FUNC_CALL(get_input2_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#endif
}

#ifdef BEAM_TABLE_TYPE
inline uint FUNC(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if BEAM_TABLE_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(BEAM_TABLE, b, f, w, z, y, x);
#else
#   error sdpa_ref.cl : Unsupported beam table format
#endif
}

inline uint FUNC(get_bt_index_key)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
}

inline uint FUNC(get_bt_index_value)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT2_DIMS_ORDER);
}
#endif

#define APPLY_SCALE_TO_QUERY 1

// query_input   [batch, heads_num, q_len, head_size]
// key_input     [batch, kv_heads_num, kv_len, head_size]
// value_input   [batch, kv_heads_num, kv_len, head_size]
// attn_mask     [1, 1, q_len, kv_len]
// output        [batch, heads_num, q_len, head_size]
// tmp_buf       [batch, heads_num, q_len, kv_len]
KERNEL(sdpa_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* query_input,
    const __global INPUT1_TYPE* key_input,
    const __global INPUT2_TYPE* value_input,
#if HAS_ATTN_MASK_INPUT
    const __global INPUT3_TYPE* attn_mask,
#endif
#if HAS_SCALE_INPUT
    const __global INPUT4_TYPE* scale,
#endif
    __global OUTPUT_TYPE* output,
#ifdef BEAM_TABLE_TYPE
    const __global BEAM_TABLE_TYPE* beam_table,
#endif
    __global OUTPUT_TYPE* tmp_buf
)
{
    // dispatchData.gws = { output.Batch().v * output.Feature().v, output.Y().v, output.X().v };
                        // = b * f, y, x

    // batch * head_num
    const uint batch_idx = get_global_id(0);

    // TransposedDimensionAccessHelperJit dims_q(params.inputs[0], params.input0_order);
    // const auto num_heads = params.conf.is_paged_attention ? std::to_string(params.conf.heads_num) : dims_q.f();
    // jit.AddConstant(MakeJitConstant("NUM_HEADS", num_heads));
    const uint b0 = batch_idx / NUM_HEADS; /* BATCH dim */
    const uint b1 = batch_idx % NUM_HEADS; /* HEADS_NUM dim */

    // jit.AddConstant(MakeJitConstant("TARGET_SEQ_LEN", dims_q.y()));
    // jit.AddConstant(MakeJitConstant("NUM_HEADS", num_heads));
    // jit.AddConstant(MakeJitConstant("NUM_KV_HEADS", params.conf.kv_heads_num));

    // q_y
    const uint target_seq_idx = get_global_id(1);
    // q_x
    const uint head_size_idx = get_global_id(2);

#if HAS_SCALE_INPUT
    const OUTPUT_TYPE scale_val = *scale;
#else
    // INPUT1_SIZE_X = head_size
    const OUTPUT_TYPE scale_val = OUTPUT_VAL_ONE / sqrt(TO_OUTPUT_TYPE(INPUT1_SIZE_X));
#endif

    // Process 1*seq_len elements (Gemm1 + SoftMax) using a single work item, saving results to tmp_buf and
    // reusing them between all work items within a single workgroup for Gemm2 calculations.
    if (get_local_id(2) == 0) {
        // jit.AddConstant(MakeJitConstant("SOURCE_SEQ_LEN", dims_k.y()));   kv_len
        for (uint s = 0; s < SOURCE_SEQ_LEN /* seq_len */; s++) {
            OUTPUT_TYPE acc = 0;
            for (uint h = 0; h < HEAD_SIZE /* head_size */; h++) {
                uint query_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0, b1, 0, 0, target_seq_idx, h);
#ifdef BEAM_TABLE_TYPE
                uint b_idx = beam_table[FUNC_CALL(get_bt_index_key)(OPTIONAL_SHAPE_INFO_TENSOR b0, b1, 0, 0, s, h)];
#else
                uint b_idx = b0;
#endif
                uint key_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1, 0, 0, s, h);

#if APPLY_SCALE_TO_QUERY
                INPUT0_TYPE q_val = query_input[query_offset] * scale_val;
#else
                INPUT0_TYPE q_val = query_input[query_offset];
#endif
                INPUT1_TYPE k_val = key_input[key_offset];
                acc += q_val * k_val;
            }

#if !APPLY_SCALE_TO_QUERY
            acc *= scale_val;
#endif

            uint tmp_buf_offset = b0 * (NUM_HEADS * TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  b1 * (TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  target_seq_idx * (SOURCE_SEQ_LEN) + s;
            tmp_buf[tmp_buf_offset] = acc;
        }

        ACCUMULATOR_TYPE qk_max = ACCUMULATOR_VAL_MIN;
        for (uint s = 0; s < SOURCE_SEQ_LEN /* seq_len */; s++) {
            uint tmp_buf_offset = b0 * (NUM_HEADS * TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  b1 * (TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  target_seq_idx * (SOURCE_SEQ_LEN) + s;
#if IS_CAUSAL
            OUTPUT_TYPE attn_mask_val = s > target_seq_idx ? OUTPUT_VAL_MIN : 0;
#elif !IS_CAUSAL && HAS_ATTN_MASK_INPUT
// #define INPUT3_GET_INDEX_SAFE(b, f, y, x) GET_DATA_INDEX_SAFE(INPUT3, b, f, y, x)
// #define GET_DATA_INDEX_SAFE(prefix, b, f, y, x) CAT(prefix, _OFFSET) + (x % CAT(prefix, _SIZE_X ))*CAT(prefix, _X_PITCH) + (y % CAT(prefix, _SIZE_Y ))*CAT(prefix, _Y_PITCH) + (f % CAT(prefix, _FEATURE_NUM))*CAT(prefix, _FEATURE_PITCH) + (b % CAT(prefix, _BATCH_NUM ))*CAT(prefix, _BATCH_PITCH)
// const uint attn_mask_offset = (start_partition_idx + seq_len % 1)*1 + (target_seq_idx + seq_idx % 1)*1 + (b1_idx % INPUT3_FEATURE_NUM) + (b0_idx % INPUT3_BATCH_NUM ))* INPUT3_FEATURE_NUM;
            uint attn_mask_offset = INPUT3_GET_INDEX_SAFE(b0, b1, target_seq_idx, s);
            OUTPUT_TYPE attn_mask_val = attn_mask[attn_mask_offset];
#else
            OUTPUT_TYPE attn_mask_val = OUTPUT_VAL_ZERO;
#endif

            OUTPUT_TYPE qk_val = tmp_buf[tmp_buf_offset] + attn_mask_val;
            tmp_buf[tmp_buf_offset] = qk_val;

            qk_max = ACCUMULATOR_MAX_FUNC(qk_max, TO_ACCUMULATOR_TYPE(qk_val));
        }

        ACCUMULATOR_TYPE exp_sum = ACCUMULATOR_VAL_ZERO;
        for (uint s = 0; s < SOURCE_SEQ_LEN /* seq_len */; s++) {
            uint tmp_buf_offset = b0 * (NUM_HEADS * TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  b1 * (TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  target_seq_idx * (SOURCE_SEQ_LEN) + s;

            OUTPUT_TYPE qk_val = tmp_buf[tmp_buf_offset];
            ACCUMULATOR_TYPE val = native_exp(TO_ACCUMULATOR_TYPE(qk_val) - qk_max);
            exp_sum += val;

            tmp_buf[tmp_buf_offset] = TO_OUTPUT_TYPE(val);
        }

        const ACCUMULATOR_TYPE inv_sum = ACCUMULATOR_VAL_ONE / exp_sum;
        for (uint s = 0; s < SOURCE_SEQ_LEN /* seq_len */; s++) {
            uint tmp_buf_offset = b0 * (NUM_HEADS * TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  b1 * (TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                                  target_seq_idx * (SOURCE_SEQ_LEN) + s;

            OUTPUT_TYPE qk_val = tmp_buf[tmp_buf_offset];
            ACCUMULATOR_TYPE val = TO_ACCUMULATOR_TYPE(qk_val) * inv_sum;
            tmp_buf[tmp_buf_offset] = TO_OUTPUT_TYPE(val);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    OUTPUT_TYPE acc = 0;
    for (uint s = 0; s < SOURCE_SEQ_LEN /* seq_len */; s++) {
        uint tmp_buf_offset = b0 * (NUM_HEADS * TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                              b1 * (TARGET_SEQ_LEN * SOURCE_SEQ_LEN) +
                              target_seq_idx * (SOURCE_SEQ_LEN) + s;

#ifdef BEAM_TABLE_TYPE
        uint b_idx = beam_table[FUNC_CALL(get_bt_index_value)(OPTIONAL_SHAPE_INFO_TENSOR b0, b1, 0, 0, s, head_size_idx)];
#else
        uint b_idx = b0;
#endif
        uint value_offset = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, b1, 0, 0, s, head_size_idx);

        acc += tmp_buf[tmp_buf_offset] * value_input[value_offset];
    }

    uint output_offset = OUTPUT_GET_INDEX(b0, b1, target_seq_idx, head_size_idx);
    output[output_offset] = acc;
}
