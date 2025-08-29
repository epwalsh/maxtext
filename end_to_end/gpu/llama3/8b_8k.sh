#!/bin/bash

set -e

BASE_OUTPUT_PATH=gs://allennlp-petew/$(date +%Y-%m-%d-%H%M%S)
export DATASET_PATH=gs://allennlp-tensorflow-datasets

export GOOGLE_APPLICATION_CREDENTIALS="$HOME/google_creds.json"
echo "$GOOGLE_CREDENTIALS" >> "$GOOGLE_APPLICATION_CREDENTIALS"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
 --xla_gpu_enable_triton_gemm=false
 --xla_gpu_enable_command_buffer=
 --xla_gpu_graph_level=0
 --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=71303168
 --xla_gpu_all_gather_combine_threshold_bytes=536870912
 --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728
 --xla_gpu_enable_pipelined_all_gather=true
 --xla_gpu_enable_pipelined_reduce_scatter=true
 --xla_gpu_enable_pipelined_all_reduce=true
 --xla_gpu_enable_while_loop_double_buffering=true
 --xla_gpu_enable_all_gather_combine_by_dim=false
 --xla_gpu_enable_reduce_scatter_combine_by_dim=false
 --xla_disable_hlo_passes=rematerialization"

# Threshold settings from mixtral config:
# --xla_gpu_all_reduce_combine_threshold_bytes=71303168
# --xla_gpu_all_gather_combine_threshold_bytes=536870912
# --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728
#
# Threshold settings from a llama 7b config:
# --xla_gpu_all_reduce_combine_threshold_bytes=134217728
# --xla_gpu_all_gather_combine_threshold_bytes=134217728
# --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864

export CUDA_DEVICE_MAX_CONNECTIONS=8
export NVTE_FUSED_ATTN=1
export NCCL_ALGO=Tree,Ring
export JAX_ENABLE_PGLE=false
export JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=true

python3 -m MaxText.train MaxText/configs/base.yml \
    model_name=llama3-8b \
    hardware=gpu \
    dataset_type=synthetic \
    base_output_directory="${BASE_OUTPUT_PATH}" \
    dataset_path="${DATASET_PATH}" \
    run_name="llama3-8b_pre_training_$(date '+%H%M%S')" \
    enable_tensorboard=false \
    async_checkpointing=false \
    attention=cudnn_flash_te \
    dtype=bfloat16 \
    enable_checkpointing=false \
    ici_fsdp_parallelism=8 \
    max_target_length=8192 \
    per_device_batch_size=4 \
    reuse_example_batch=1 \
    steps=120 \
    tokenizer_path=assets/tokenizer_llama3.tiktoken \
    tokenizer_type=tiktoken \
    weight_dtype=bfloat16 \
    sparse_matmul=False \
    packing=False \
    remat_policy=minimal_with_context

    # quantization=fp8

echo "Finished pre-training"
