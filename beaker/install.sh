#!/bin/bash

set -e

pip uninstall -y MaxText
uv pip install --system --break-system-packages -r beaker/requirements.txt
