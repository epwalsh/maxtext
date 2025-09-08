#!/usr/bin/env bash

set -e

echo "Patching fused ring attention..."
wget https://gist.githubusercontent.com/epwalsh/8fbde5374638b62f49743a219831dc7c/raw/4c3dabe6d6ea51534fdc20ad6debaf1347fd7961/patched_attention.py
mv patched_attention.py /usr/local/lib/python3.12/dist-packages/transformer_engine/jax/cpp_extensions/attention.py
wget https://gist.githubusercontent.com/epwalsh/94d6a0d506dcae906419df89b0e53ab2/raw/c517f112e2bb46f5ff306bc0aa7c3f466076d551/transformer.py
mv transformer.py /usr/local/lib/python3.12/dist-packages/transformer_engine/jax/flax/transformer.py
