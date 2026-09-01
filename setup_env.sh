#!/bin/bash
# DeGibbs Virtual Environment Setup for Linux
# Run this script from the project root directory

set -e

echo -e "\033[32mCreating Python virtual environment...\033[0m"
python3 -m venv .venv

echo -e "\033[32mActivating virtual environment...\033[0m"
source .venv/bin/activate

echo -e "\033[32mUpgrading pip...\033[0m"
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 (A6000 compatible)
echo -e "\033[32mInstalling PyTorch with CUDA support...\033[0m"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install the package in development mode
echo -e "\033[32mInstalling DeGibbs package...\033[0m"
pip install -e .

# Install development dependencies
echo -e "\033[32mInstalling development dependencies...\033[0m"
pip install -e ".[dev]"

echo -e "\n\033[32mSetup complete!\033[0m"
echo -e "\033[33mTo activate the environment, run: source .venv/bin/activate\033[0m"
echo -e "\033[33mTo train on 2 GPUs, run:\033[0m"
echo -e "\033[33m  python train.py --data_dir /path/to/imagenet --devices 2 --batch_size 32\033[0m"
