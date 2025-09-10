#!/usr/bin/env bash

set -e

transformer_engine_location=$(python -c 'import transformer_engine; print(transformer_engine.__file__)')
transformer_engine_location=$(dirname "$transformer_engine_location")
echo "Patching fused ring attention to ${transformer_engine_location}..."

# wget https://gist.githubusercontent.com/epwalsh/8fbde5374638b62f49743a219831dc7c/raw/4c3dabe6d6ea51534fdc20ad6debaf1347fd7961/patched_attention.py
#wget https://gist.githubusercontent.com/epwalsh/8fbde5374638b62f49743a219831dc7c/raw/37d63962d28f746847ecace93e2e79a945cf5775/patched_attention.py
#mv patched_attention.py "${transformer_engine_location}/jax/cpp_extensions/attention.py"

wget https://gist.githubusercontent.com/epwalsh/94d6a0d506dcae906419df89b0e53ab2/raw/c517f112e2bb46f5ff306bc0aa7c3f466076d551/transformer.py
mv transformer.py "${transformer_engine_location}/jax/flax/transformer.py"
