#!/bin/bash

set -e

BASE_OUTPUT_PATH=gs://allennlp-petew/$(date +%Y-%m-%d-%H%M%S)
export DATASET_PATH=gs://allennlp-tensorflow-datasets

export GOOGLE_APPLICATION_CREDENTIALS="$HOME/google_creds.json"
echo "$GOOGLE_CREDENTIALS" >> "$GOOGLE_APPLICATION_CREDENTIALS"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
 --xla_gpu_enable_triton_gemm=false
 --xla_gpu_graph_level=0 --xla_gpu_enable_highest_priority_async_stream=true
 --xla_gpu_all_reduce_combine_threshold_bytes=134217728 --xla_gpu_all_gather_combine_threshold_bytes=134217728
 --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=true
 --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true
 --xla_gpu_enable_while_loop_double_buffering=true
 --xla_gpu_enable_all_gather_combine_by_dim=false
 --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization"

# `SCANNED_CHECKPOINT` refers to the checkpoint that used for both `train.py` and `decode.py` 
# if [ -z "${SCANNED_CHECKPOINT}" ]; then
#     # Non-Googlers please remember to point SCANNED_CHECKPOINT to GCS buckets that you own
#     export SCANNED_CHECKPOINT=${BASE_OUTPUT_PATH}/8x7/scanned_ckpt/0/items
#     echo "SCANNED_CHECKPOINT is not set, using BASE_OUTPUT_PATH = ${SCANNED_CHECKPOINT}"
# fi

# `UNSCANNED_CHECKPOINT` refers to run decoding
# if [ -z "${UNSCANNED_CKPT_PATH}" ]; then
#     # Non-Googlers please remember to point UNSCANNED_CKPT_PATH to GCS buckets that you own
#     export UNSCANNED_CKPT_PATH=${BASE_OUTPUT_PATH}/unscanned_ckpt/checkpoints/0/items
#     echo "UNSCANNED_CKPT_PATH is not set, using BASE_OUTPUT_PATH = ${UNSCANNED_CKPT_PATH}"
# fi


# Run pre-training - dropping implementation
python3 -m MaxText.train MaxText/configs/base.yml model_name=mixtral-8x7b hardware=gpu \
    dataset_type=synthetic \
    base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
    run_name="dropping_pre_training_$(date '+%H%M%S')" async_checkpointing=false \
    attention=cudnn_flash_te capacity_factor=1.0 dtype=bfloat16 \
    enable_checkpointing=false ici_expert_parallelism=-1 ici_fsdp_parallelism=1 \
    max_target_length=4096 megablox=False per_device_batch_size=2 \
    reuse_example_batch=1 steps=100 tokenizer_path=assets/tokenizer.mistral-v1 \
    weight_dtype=bfloat16 sparse_matmul=False packing=False \
    remat_policy=minimal quantization=fp8
echo "Finished pre-training"

# Run fine-tuning - dropping implementation
# python3 -m MaxText.train MaxText/configs/base.yml model_name=mixtral-8x7b hardware=gpu \
#     load_parameters_path=${SCANNED_CHECKPOINT} \
#     base_output_directory=${BASE_OUTPUT_PATH} dataset_path=${DATASET_PATH} \
#     run_name=dropping_pre_training async_checkpointing=true \
#     attention=cudnn_flash_te capacity_factor=1.25 dtype=bfloat16 \
#     ici_expert_parallelism=-1 ici_fsdp_parallelism=1 \
#     max_target_length=1024 megablox=False per_device_batch_size=1 \
#     reuse_example_batch=1 steps=5 tokenizer_path=assets/tokenizer.mistral-v1 \
#     weight_dtype=bfloat16 sparse_matmul=False packing=False
# echo "Finished fine-tuning"

# # TODO(b/391864113): Add this once the bug is fixed
# # Run decoding with converted ckpt - dropping implementation
# python3 -m MaxText.decode MaxText/configs/base.yml model_name=mixtral-8x7b hardware=gpu \
#     run_name=unscanned_decoding load_parameters_path=${UNSCANNED_CKPT_PATH} \
#     async_checkpointing=false attention=dot_product capacity_factor=0.1 \
#     ici_expert_parallelism=8 ici_fsdp_parallelism=1 max_prefill_predict_length=11 \
#     max_target_length=24 megablox=False per_device_batch_size=1 \
#     prompt='"[INST] I love to [/INST]"' scan_layers=false \
#     tokenizer_path=assets/tokenizer.mistral-v1
# echo "Finished decoding"


