#!/bin/bash

################################################################################
# TinyAI Training Job Script for OSCAR
#
# This script submits training jobs for both the control transformer and
# the recursive (TRM-inspired) transformer.
#
# Usage: sbatch train_job.sh
#        sbatch train_job.sh --skip_control    # Only train recursive
#        sbatch train_job.sh --skip_recursive  # Only train control
################################################################################

#SBATCH --job-name=tinyai_train
#SBATCH --output=output/logs/slurm_%j.out
#SBATCH --error=output/logs/slurm_%j.err

# GPU Configuration
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Note: Removed --constraint=a100 to use any available GPU (faster queue time)
# Your model is small enough to run on any GPU type

# Resource Configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Time Configuration (adjust based on your needs)
#SBATCH --time=04:00:00

# Email notifications (optional - uncomment and set your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@brown.edu

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "TinyAI Training Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=============================================="

# Load conda
module load miniconda3/23.11.0s
source activate tinyai

# Verify GPU is available
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Verify TensorFlow can see the GPU
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow sees {len(gpus)} GPU(s)')"
echo ""

# ============================================================================
# Create output directories
# ============================================================================
mkdir -p output/logs
mkdir -p output/checkpoints/control
mkdir -p output/checkpoints/recursive

# ============================================================================
# Training Configuration
# ============================================================================

# Model hyperparameters
D_MODEL=256
NUM_LAYERS=2
NUM_HEADS=4
FF_DIM=512
DROPOUT=0.1

# Recursive model specific
DEEP_REC_CYCLES=3
NUM_L_STEPS=6
DEEP_SUP_STEPS=4
ACT_LOSS_WEIGHT=0.1

# Training hyperparameters
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=1e-4
MAX_SEQ_LENGTH=256

# Paths
DATA_PATH="preprocessing/data/final_train_data.txt"
VOCAB_PATH="preprocessing/data/vocab.json"
OUTPUT_DIR="output"

# Parse command line arguments passed through sbatch
EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

# ============================================================================
# Run Training
# ============================================================================

echo "=============================================="
echo "Starting Training"
echo "=============================================="
echo "Configuration:"
echo "  d_model: $D_MODEL"
echo "  num_layers: $NUM_LAYERS"
echo "  num_heads: $NUM_HEADS"
echo "  ff_dim: $FF_DIM"
echo "  deep_rec_cycles: $DEEP_REC_CYCLES"
echo "  num_l_steps: $NUM_L_STEPS"
echo "  epochs: $EPOCHS"
echo "  batch_size: $BATCH_SIZE"
echo "  learning_rate: $LEARNING_RATE"
echo "=============================================="
echo ""

# Run the training script
python train.py \
    --data_path "$DATA_PATH" \
    --vocab_path "$VOCAB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --ff_dim $FF_DIM \
    --dropout_rate $DROPOUT \
    --deep_rec_cycles $DEEP_REC_CYCLES \
    --num_l_steps $NUM_L_STEPS \
    --deep_sup_steps $DEEP_SUP_STEPS \
    --act_loss_weight $ACT_LOSS_WEIGHT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    $EXTRA_ARGS

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "=============================================="
echo "Training Complete"
echo "End Time: $(date)"
echo "=============================================="

# Print final results if available
if [ -f "output/training_results.json" ]; then
    echo ""
    echo "Training Results:"
    cat output/training_results.json
fi

