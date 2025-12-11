#!/bin/bash

#SBATCH --nodes=1               # node count
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 04:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=64000MB           # 64GB memory
#SBATCH --job-name='TinyAI'
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err

# Force unbuffered output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# TensorFlow GPU settings
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load conda environment
module load miniconda3/23.11.0s

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate environment
conda activate tinyai

# Verify environment is active and Python path
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Install/verify required packages
echo "Installing/verifying required packages..."
pip install --quiet --upgrade pip
pip install --quiet tensorflow==2.15.0 numpy tqdm datasets openai wandb

# Verify TensorFlow can be imported
python -c "import tensorflow as tf; import numpy as np; print(f'TensorFlow {tf.__version__} and NumPy {np.__version__} imported successfully')" || {
    echo "ERROR: Failed to import TensorFlow or NumPy"
    exit 1
}

# Check GPU allocation
echo "GPU Information:"
nvidia-smi
echo ""

# Test TensorFlow GPU detection
echo "TensorFlow GPU Detection:"
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPUs detected:', len(tf.config.list_physical_devices('GPU'))); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
echo ""

# Create output directories
mkdir -p slurm_logs
mkdir -p output/checkpoints/control
mkdir -p output/checkpoints/recursive

# ============================================================================
# Training Configuration
# ============================================================================

# Model hyperparameters (leaner for recursive stability)
D_MODEL=32
NUM_LAYERS=1
NUM_HEADS=2
FF_DIM=32
DROPOUT=0.1

# Recursive model specific (reduced to lower memory)
DEEP_REC_CYCLES=2 #2
NUM_L_STEPS=3 #3
DEEP_SUP_STEPS=2
ACT_LOSS_WEIGHT=0.1
STEP_PENALTY_WEIGHT=0.01

# Training hyperparameters (smaller batch/seq to reduce RAM)
EPOCHS=30
BATCH_SIZE=8
LEARNING_RATE=1e-3
MAX_SEQ_LENGTH=128

# Paths
DATA_PATH="preprocessing/data/final_train_data.txt"
VOCAB_PATH="preprocessing/data/vocab.json"
OUTPUT_DIR="output"

# Parse command line arguments passed through sbatch
EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

echo "=========================================="
echo "Starting main Python script at $(date)"
echo "=========================================="
echo ""
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
echo "  max_seq_length: $MAX_SEQ_LENGTH"
echo ""
echo "To monitor training with WandB:"
echo "  1. First, log in to WandB (one time only):"
echo "     python wandb_login.py"
echo "  2. Then view metrics at: https://wandb.ai"
echo "  (No port forwarding needed - metrics upload automatically!)"
echo ""

# Run the training script
python -u train.py \
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
    --step_penalty_weight $STEP_PENALTY_WEIGHT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Python script finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Print final results if available
if [ -f "output/training_results.json" ]; then
    echo ""
    echo "Training Results:"
    cat output/training_results.json
fi

exit $EXIT_CODE

