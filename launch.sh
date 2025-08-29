#!/bin/bash

set -ex

script="${1:-mixtral/8x7b_8k.sh}"
#script="${1:-llama3/8b_8k.sh}"
#script="${1:-llama3/70b_8k.sh}"

# Remove file extension for naming.
name="${script%.*}"
# Replace slashes in path with dashes.
name="${name//\//-}"
# Replace underscores with dashes.
name="${name//_/-}"

gantry run \
    --show-logs \
    --yes \
    --allow-dirty \
    --name="${name}-$(date +%Y%m%d-%H%M%S)" \
    --description="MaxText ${name}" \
    --group=petew/B200_benchmarks \
    --group="petew/B200_benchmarks_${name}" \
    --priority=urgent \
    --task-timeout=20m \
    --env-secret='GOOGLE_CREDENTIALS=GOOGLE_CREDENTIALS' \
    --env-secret='BEAKER_TOKEN' \
    --beaker-image=petew/olmax \
    --system-python \
    --gpu-type=b200 \
    --gpus=8 -- \
    "./end_to_end/gpu/${script}"
