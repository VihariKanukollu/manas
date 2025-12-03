#!/bin/bash
set -e

echo "=== Upgrading pip, wheel, setuptools ==="
pip install --upgrade pip wheel setuptools

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Reinstalling adam-atan2 with --no-build-isolation ==="
pip uninstall -y adam-atan2
pip install --no-cache-dir --no-build-isolation adam-atan2

echo "=== Logging into Weights & Biases ==="
wandb login 1dc0a6d6c0fcfbea7784e7923daf7aba591ac301

echo "=== Setup complete ==="


