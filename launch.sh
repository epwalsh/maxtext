#!/bin/bash

set -ex

gantry run \
    --show-logs \
    --yes \
    --workspace=ai2/OLMo-pretraining-stability \
    --group=petew/B200_benchmarks \
    --group=petew/B200_benchmarks_mixtral \
    --priority=high \
    --allow-dirty \
    --env-secret='GOOGLE_CREDENTIALS=GOOGLE_CREDENTIALS' \
    --beaker-image=petew/olmax \
    --system-python \
    --gpu-type=b200 \
    --gpus=8 -- \
    ./end_to_end/gpu/mixtral/test_8x7b.sh
