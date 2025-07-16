#!/bin/bash

set -ex

gantry run \
    --show-logs \
    --yes \
    --workspace=ai2/OLMo-pretraining-stability \
    --priority=high \
    --allow-dirty \
    --env-secret='GOOGLE_CREDENTIALS=GOOGLE_CREDENTIALS' \
    --beaker-image=petew/olmax \
    --system-python \
    --gpu-type=b200 \
    --gpus=8 -- \
    ./end_to_end/gpu/mixtral/test_8x7b.sh
