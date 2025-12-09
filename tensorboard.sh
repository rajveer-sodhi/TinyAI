#!/bin/bash
################################################################################
# TensorBoard Helper Script for OSCAR
#
# This script runs TensorBoard on OSCAR with port forwarding support.
# Use this to view training metrics DURING training.
#
# Usage:
#   1. First, set up port forwarding on your LOCAL machine:
#      ssh -L 6006:localhost:6006 your_username@ssh.oscar.brown.edu
#      (Keep this terminal open!)
#
#   2. Then, on OSCAR, run this script:
#      bash tensorboard.sh
#
#   3. Open http://localhost:6006 in your browser
#
# Note: This script will request an interactive GPU session if you're not
# already on a compute node. If you're already on a compute node, it will
# run TensorBoard directly.
################################################################################

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================="
echo "TensorBoard Server for TinyAI"
echo -e "==============================================${NC}"
echo ""

# Check if we're on a compute node (has GPU)
if command -v nvidia-smi &> /dev/null; then
    # We're on a compute node, run TensorBoard directly
    echo -e "${GREEN}Detected GPU node. Running TensorBoard directly...${NC}"
    echo ""
    
    # Load environment
    module load miniconda3/23.11.0s 2>/dev/null || true
    source activate tinyai 2>/dev/null || conda activate tinyai 2>/dev/null
    
    # Install tensorboard if not already installed
    pip install tensorboard -q 2>/dev/null
    
    # Check if log directory exists
    if [ ! -d "output/logs" ]; then
        echo -e "${RED}Error: output/logs directory not found!${NC}"
        echo "Make sure you're in the TinyAI project directory and training has started."
        exit 1
    fi
    
    # Get the node name
    NODE=$(hostname)
    echo -e "${GREEN}Running TensorBoard on node: $NODE${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT:${NC}"
    echo "Before running this script, make sure you've set up port forwarding"
    echo "on your LOCAL machine with:"
    echo -e "  ${BLUE}ssh -L 6006:localhost:6006 your_username@ssh.oscar.brown.edu${NC}"
    echo ""
    echo "Then open http://localhost:6006 in your browser"
    echo ""
    echo -e "${GREEN}Starting TensorBoard...${NC}"
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Run TensorBoard
    tensorboard --logdir=output/logs --host=0.0.0.0 --port=6006 --reload_interval=5
    
else
    # We're on a login node, need to request interactive session
    echo -e "${YELLOW}You're on a login node. Requesting interactive GPU session...${NC}"
    echo ""
    echo "This will request a 2-hour interactive session."
    echo "Press Ctrl+C to cancel, or wait for the session to start..."
    echo ""
    
    # Request interactive session and run TensorBoard
    srun --partition=gpu --gres=gpu:1 --time=02:00:00 --pty bash << 'EOF'
        # Load conda and activate environment
        module load miniconda3/23.11.0s
        source activate tinyai
        
        # Install tensorboard if not already installed
        pip install tensorboard -q
        
        # Check if log directory exists
        if [ ! -d "output/logs" ]; then
            echo "Error: output/logs directory not found!"
            echo "Make sure you're in the TinyAI project directory and training has started."
            exit 1
        fi
        
        # Get the node name
        NODE=$(hostname)
        echo ""
        echo "=============================================="
        echo "TensorBoard Server"
        echo "=============================================="
        echo "Running on node: $NODE"
        echo "Log directory: output/logs"
        echo ""
        echo "IMPORTANT:"
        echo "Before running this script, make sure you've set up port forwarding"
        echo "on your LOCAL machine with:"
        echo "  ssh -L 6006:localhost:6006 your_username@ssh.oscar.brown.edu"
        echo ""
        echo "Then open http://localhost:6006 in your browser"
        echo "=============================================="
        echo ""
        echo "Starting TensorBoard..."
        echo "Press Ctrl+C to stop"
        echo ""
        
        # Run TensorBoard
        tensorboard --logdir=output/logs --host=0.0.0.0 --port=6006 --reload_interval=5
EOF
fi

