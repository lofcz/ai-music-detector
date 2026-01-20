#!/bin/bash
# AI Music Detector - Linux/macOS Environment Setup
# This script creates a conda environment with all dependencies

echo "============================================"
echo "AI Music Detector - Environment Setup"
echo "============================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda not found. Please install Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Found conda installation"
echo ""

# Initialize conda for script
eval "$(conda shell.bash hook)"

# Check if environment already exists
if conda env list | grep -q "ai-music-detector"; then
    echo "Environment 'ai-music-detector' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " OVERWRITE
    if [[ "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ai-music-detector -y
    else
        echo "Skipping environment creation."
        echo ""
        echo "To activate: conda activate ai-music-detector"
        exit 0
    fi
fi

echo "Creating conda environment..."
echo "This may take several minutes..."
echo ""

conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Failed to create full environment."
    echo "Trying alternative setup with minimal conda + pip..."
    
    conda create -n ai-music-detector python=3.11 -y
    conda activate ai-music-detector
    pip install -r requirements.txt
fi

echo ""
echo "============================================"
echo "Environment created successfully!"
echo "============================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ai-music-detector"
echo ""
echo "Then you can run the training pipeline:"
echo "  python download_data.py --dataset all"
echo "  python extract_fakeprints.py --input ... --output ... --label ..."
echo "  python train_model.py --real ... --fake ..."
echo "  python export_onnx.py"
echo ""
