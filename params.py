#!/usr/bin/env python3
import os, sys, json
import tensorflow as tf

sys.path.insert(0, 'components')
from transformer import Transformer
import importlib.util
spec = importlib.util.spec_from_file_location("rt", "components/recursive-transformer.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
RecursiveTransformer = rt.RecursiveTransformer

# Config from train.sh (current)
vocab_size = len(json.load(open("preprocessing/data/vocab.json")))
D_MODEL = 128
NUM_LAYERS = 2
NUM_HEADS = 4
FF_DIM = 256
DROPOUT = 0.1
MAX_SEQ_LENGTH = 128

# Recursive-specific
DEEP_REC_CYCLES = 4      # Multiple cycles for deep iterative refinement
NUM_L_STEPS = 6          # More inner iterations per cycle
DEEP_SUP_STEPS = 4       # More supervision steps for better learning
ACT_LOSS_WEIGHT = 0.01   # Lower weight to reduce interference
STEP_PENALTY_WEIGHT = 0.001
MIN_HALT_STEPS = 3
STAGE_WEIGHTS = "1.0,1.5,2.0,2.5"  # Emphasize later refinement steps
HALT_EXPLORATION_PROB = 0.1
HALT_MAX_STEPS = 16

dummy = tf.zeros((1, MAX_SEQ_LENGTH), dtype=tf.int32)

# Build models
control = Transformer(vocab_size, D_MODEL, MAX_SEQ_LENGTH, NUM_LAYERS, NUM_HEADS, FF_DIM, DROPOUT)
control(dummy)

recursive = RecursiveTransformer(
    vocab_size, D_MODEL, MAX_SEQ_LENGTH, NUM_LAYERS, NUM_HEADS, FF_DIM, DROPOUT,
    DEEP_REC_CYCLES, DEEP_SUP_STEPS, ACT_LOSS_WEIGHT, NUM_L_STEPS, HALT_EXPLORATION_PROB, HALT_MAX_STEPS,
    STEP_PENALTY_WEIGHT, MIN_HALT_STEPS, STAGE_WEIGHTS
)
recursive(dummy)

# Count params
print(f"Control:   {control.count_params():,} ({control.count_params()/1e6:.2f}M)")
print(f"Recursive: {recursive.count_params():,} ({recursive.count_params()/1e6:.2f}M)")
print(f"Diff:      {recursive.count_params() - control.count_params():,}")

