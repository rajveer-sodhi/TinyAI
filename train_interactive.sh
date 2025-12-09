#!/bin/bash

################################################################################
# TinyAI Interactive Training Script for OSCAR
#
# Use this for debugging and testing on an interactive GPU session.
#
# First, get an interactive GPU session:
#   interact -g 1 -m 32g -t 02:00:00
#
# Then run this script:
#   bash train_interactive.sh
#
# Or for a quick test with fewer epochs:
#   bash train_interactive.sh --epochs 2
################################################################################

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=============================================="
echo "TinyAI Interactive Training"
echo -e "==============================================${NC}"

# Check if on a GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}WARNING: nvidia-smi not found. You may not be on a GPU node.${NC}"
    echo "To get an interactive GPU session, run:"
    echo "  interact -g 1 -m 32g -t 02:00:00"
    echo ""
fi

# Load environment
echo -e "${BLUE}Loading environment...${NC}"
module load miniconda3/23.11.0s
source activate tinyai

# Verify GPU
echo ""
echo -e "${BLUE}GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow sees {len(gpus)} GPU(s)')"
echo ""

# Create output directories
mkdir -p output/logs
mkdir -p output/checkpoints/control
mkdir -p output/checkpoints/recursive

# Default to fewer epochs for interactive testing
EPOCHS=${EPOCHS:-5}

echo -e "${GREEN}Starting training with $EPOCHS epochs...${NC}"
echo ""

# Run training with any additional arguments passed to this script
python train.py \
    --epochs $EPOCHS \
    --batch_size 16 \
    --max_seq_length 128 \
    "$@"

echo ""
echo -e "${GREEN}Training complete!${NC}"

