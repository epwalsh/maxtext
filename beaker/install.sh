#!/bin/bash

set -e

pip uninstall -y MaxText
uv pip install -r beaker/requirements.txt
