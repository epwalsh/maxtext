#!/bin/bash

set -ex

script="${1:-beaker/llama3_8b_8k.sh}"
num_nodes="${2:-1}"

GANTRY_MULTI_NODE_ARGS=()
if (( num_nodes > 1 )); then
    GANTRY_MULTI_NODE_ARGS+=(
        --replicas="$num_nodes"
        --leader-selection
        --host-networking
        --propagate-failure
        --propagate-preemption
        --synchronized-start-timeout='5m'
    )
fi

name=$(basename "$script")
# Remove file extension for naming.
name="${name%.*}"
# Replace slashes in path with dashes.
name="${name//\//-}"
# Replace underscores with dashes for run name.
name="${name//_/-}"
# Keep group name with underscores.
group_name="${name//-/_}"

gantry run \
    --show-logs \
    --yes \
    --name="${name}-$(date +%Y%m%d-%H%M%S)" \
    --allow-dirty \
    --workspace=ai2/google_benchmarks \
    --description="MaxText ${name}" \
    --group=petew/B200_benchmarks \
    --group="petew/B200_benchmarks_${group_name}" \
    --priority=urgent \
    --task-timeout=120m \
    --slack-webhook-url="$SLACK_WEBHOOK_URL" \
    --env-secret='GOOGLE_CREDENTIALS=GOOGLE_CREDENTIALS' \
    --env-secret='BEAKER_TOKEN' \
    --beaker-image=petew/maxtext \
    --system-python \
    --post-setup=beaker/patch_fused_attention.sh \
    "${GANTRY_MULTI_NODE_ARGS[@]}" \
    --gpu-type=h100 \
    --gpus=8 -- \
    "./end_to_end/gpu/${script}"

    # --replicas=2 \
    # --leader-selection \
    # --host-networking \
    # --propagate-failure \
    # --propagate-preemption \
    # --synchronized-start-timeout='5m' \
    #
    # --beaker-image=petew/maxtext \
    # --install=beaker/install.sh \
    # --beaker-image=petew/olmax-25.08 \
