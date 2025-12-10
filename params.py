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

# Config from train.sh
vocab_size = len(json.load(open("preprocessing/data/vocab.json")))
dummy = tf.zeros((1, 256), dtype=tf.int32)

# Build models
control = Transformer(vocab_size, 256, 256, 2, 4, 512, 0.1)
control(dummy)

recursive = RecursiveTransformer(vocab_size, 256, 256, 2, 4, 512, 0.1, 3, 4, 0.1, 6, 0.1, 16)
recursive(dummy)

# Count params
print(f"Control:   {control.count_params():,} ({control.count_params()/1e6:.2f}M)")
print(f"Recursive: {recursive.count_params():,} ({recursive.count_params()/1e6:.2f}M)")
print(f"Diff:      {recursive.count_params() - control.count_params():,}")

