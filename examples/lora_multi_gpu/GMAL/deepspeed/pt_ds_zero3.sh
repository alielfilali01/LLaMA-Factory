#!/bin/bash

NPROC_PER_NODE=4
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/train.py examples/lora_multi_gpu/GMAL/configs/tests/test_multi_gpu_push_hub.yaml
    # --push_to_hub True \
    # --hub_model_id username/model \
    # --hub_strategy checkpoint \
    # --hub_token xxx \
    # --hub_private_repo True \
