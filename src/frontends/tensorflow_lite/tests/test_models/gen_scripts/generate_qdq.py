# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Model:
# (float32)tensor_0[12] -> QUANTIZE -> (int8)tensor[12] -> DEQUANTIZE -> (float32)tensor_1[12]
# QUANTIZE:
#   scale: 0.25
#   zero point: 16
sm_path = os.path.join(sys.argv[1], "qdq_int8")
tflite_model = b'\x0c\x00\x00\x00TFL3\x00\x00\x00\x00\xba\xff\xff\xff\x03\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x1c\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\xe8\x01\x00\x00\xc8\x01\x00\x00\x01\x00\x00\x00,\x00\x00\x00\n\x00\x00\x00test_model\x00\x00\x01\x00\x00\x00l\x00\x00\x00\x00\x00\x0e\x00\x18\x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x0e\x00\x00\x00\x14\x00\x00\x00 \x00\x00\x00$\x00\x00\x00(\x00\x00\x000\x00\x00\x00\x03\x00\x00\x00T\x01\x00\x00\xe8\x00\x00\x00\xb4\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00p\x00\x00\x00,\x00\x00\x00\r\x00\x00\x00test_subgraph\x00\x00\x00\xc4\xff\xff\xff\x00\x00\x0e\x00\x18\x00\x08\x00\x0c\x00\x10\x00\x07\x00\x14\x00\x0e\x00\x00\x00\x00\x00\x00&\x01\x00\x00\x00\x0c\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x06\x00\x04\x00\x00\x00\x00\x00\x0e\x00\x14\x00\x00\x00\x08\x00\x0c\x00\x07\x00\x10\x00\x0e\x00\x00\x00\x00\x00\x00Y\x0c\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x04\x00\x04\x00\x04\x00\x00\x00t\xff\xff\xff\x08\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x00\x00tensor_2\x00\x00\x0e\x00\x14\x00\x08\x00\x07\x00\x00\x00\x0c\x00\x10\x00\x0e\x00\x00\x00\x00\x00\x00\t\x0c\x00\x00\x00\x10\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x00\x00tensor_1\x00\x00\x00\x00\x0c\x00\x0c\x00\x00\x00\x00\x00\x04\x00\x08\x00\x0c\x00\x00\x00\x08\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x00\x00\x80>\x01\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x0c\x00\x04\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x00\x00\x08\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x00\x00tensor_0\x00\x00\x00\x00\xf0\xff\xff\xff\x00\x00\x00\x06\x03\x00\x00\x00\x06\x00\x00\x00\x0c\x00\x10\x00\x07\x00\x00\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x00\x00r\x03\x00\x00\x00r\x00\x00\x00'
with open(os.path.join(sys.argv[1], sm_path + ".tflite"), 'wb') as f:
  f.write(tflite_model)

# Model:
# (float32)tensor_0[12] -> QUANTIZE -> (uint8)tensor[12] -> DEQUANTIZE -> (float32)tensor_1[12]
# QUANTIZE:
#   scale: 0.25
#   zero point: 16
sm_path = os.path.join(sys.argv[1], "qdq_uint8")
tflite_model = b'\x0c\x00\x00\x00TFL3\x00\x00\x00\x00\xba\xff\xff\xff\x03\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x1c\x00\x00\x00(\x00\x00\x00\x02\x00\x00\x00\xe8\x01\x00\x00\xc8\x01\x00\x00\x01\x00\x00\x00,\x00\x00\x00\n\x00\x00\x00test_model\x00\x00\x01\x00\x00\x00l\x00\x00\x00\x00\x00\x0e\x00\x18\x00\x04\x00\x08\x00\x0c\x00\x10\x00\x14\x00\x0e\x00\x00\x00\x14\x00\x00\x00 \x00\x00\x00$\x00\x00\x00(\x00\x00\x000\x00\x00\x00\x03\x00\x00\x00T\x01\x00\x00\xe8\x00\x00\x00\xb4\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00p\x00\x00\x00,\x00\x00\x00\r\x00\x00\x00test_subgraph\x00\x00\x00\xc4\xff\xff\xff\x00\x00\x0e\x00\x18\x00\x08\x00\x0c\x00\x10\x00\x07\x00\x14\x00\x0e\x00\x00\x00\x00\x00\x00&\x01\x00\x00\x00\x0c\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x04\x00\x06\x00\x04\x00\x00\x00\x00\x00\x0e\x00\x14\x00\x00\x00\x08\x00\x0c\x00\x07\x00\x10\x00\x0e\x00\x00\x00\x00\x00\x00Y\x0c\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x04\x00\x04\x00\x04\x00\x00\x00t\xff\xff\xff\x08\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x00\x00tensor_2\x00\x00\x0e\x00\x14\x00\x08\x00\x07\x00\x00\x00\x0c\x00\x10\x00\x0e\x00\x00\x00\x00\x00\x00\x03\x0c\x00\x00\x00\x10\x00\x00\x00(\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x00\x00tensor_1\x00\x00\x00\x00\x0c\x00\x0c\x00\x00\x00\x00\x00\x04\x00\x08\x00\x0c\x00\x00\x00\x08\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x00\x00\x80>\x01\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x0c\x00\x04\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x00\x00\x08\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x08\x00\x00\x00tensor_0\x00\x00\x00\x00\xf0\xff\xff\xff\x00\x00\x00\x06\x03\x00\x00\x00\x06\x00\x00\x00\x0c\x00\x10\x00\x07\x00\x00\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x00\x00r\x03\x00\x00\x00r\x00\x00\x00'
with open(os.path.join(sys.argv[1], sm_path + ".tflite"), 'wb') as f:
  f.write(tflite_model)
