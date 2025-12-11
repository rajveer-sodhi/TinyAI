"""
TinyAI Training Script

Training loops for control transformer and recursive transformer.
Implements comparison between single-pass and TRM-inspired recursive reasoning.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# WandB for experiment tracking
import wandb


# Add components directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))

from transformer import Transformer
from embeddings import EmbeddingLayer
from transformer_encoder import TransformerEncoder
from data.tokenizer import Tokenizer

# Import RecursiveTransformer - note the filename uses hyphen
import importlib.util
recursive_transformer_path = os.path.join(os.path.dirname(__file__), 'components', 'recursive-transformer.py')
spec = importlib.util.spec_from_file_location("recursive_transformer", recursive_transformer_path)
recursive_transformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recursive_transformer_module)
RecursiveTransformer = recursive_transformer_module.RecursiveTransformer

def load_data(q_data_path, a_data_path, input_tokenizer, ans_tokenizer, max_seq_length=512, test_split=0.1, val_split=0.1):
    """Load and tokenize data from text files."""
    with open(q_data_path, 'r', encoding='utf-8') as f:
        q_lines = [line.strip() for line in f if line.strip()]
    with open(a_data_path, 'r', encoding='utf-8') as f:
        a_lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(q_lines)} questions")
    print(f"Loaded {len(a_lines)} answers")
    
    if len(q_lines) != len(a_lines):
        raise ValueError("Mismatch in quantity of questions and answers!")
    
    encoded_q = []
    encoded_a = []
    
    for q, a in zip(q_lines, a_lines):
        q_tokens = input_tokenizer.encode(q, max_length=max_seq_length, padding=True)
        a_tokens = ans_tokenizer.encode(a, max_length=max_seq_length, padding=True)
        
        encoded_q.append(q_tokens)
        encoded_a.append(a_tokens)
    
    encoded_q = np.array(encoded_q, dtype=np.int32)
    encoded_a = np.array(encoded_a, dtype=np.int32)
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(encoded_q))
    encoded_q = encoded_q[indices]
    encoded_a = encoded_a[indices]
    
    n_total = len(encoded_q)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_test - n_val

    q_train = encoded_q[:n_train]
    a_train = encoded_a[:n_train]

    q_val = encoded_q[n_train:n_train+n_val]
    a_val = encoded_a[n_train:n_train+n_val]

    q_test = encoded_q[n_train+n_val:]
    a_test = encoded_a[n_train+n_val:]
    
    print(f"Train: {len(q_train)} | Val: {len(q_val)} | Test: {len(q_test)}")
    
    return {
        "train": (q_train, a_train),
        "val": (q_val, a_val),
        "test": (q_test, a_test),
        "input_tokenizer": input_tokenizer,
        "ans_tokenizer": ans_tokenizer,
    }


def create_tf_dataset(q_array, a_array, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((q_array, a_array))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(q_array))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.accuracies = []
    
    def update(self, loss, accuracy):
        self.losses.append(float(loss))
        self.accuracies.append(float(accuracy))
    
    @property
    def avg_loss(self):
        return np.mean(self.losses) if self.losses else 0.0
    
    @property
    def avg_accuracy(self):
        return np.mean(self.accuracies) if self.accuracies else 0.0


def compute_accuracy(logits, targets, pad_token_id):
    """Compute accuracy ignoring padding tokens."""
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    mask = tf.cast(targets != pad_token_id, tf.float32)
    correct = tf.cast(predictions == targets, tf.float32) * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)


def compute_loss_with_mask(logits, targets, pad_token_id):
    """Compute cross-entropy loss ignoring padding tokens."""
    mask = tf.cast(targets != pad_token_id, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def compute_deep_supervision_loss(intermediate_logits, targets, pad_token_id, decay_factor=0.8):
    """Compute deep supervision loss with exponential decay weighting."""
    total_loss = 0.0
    num_outputs = len(intermediate_logits)
    for i, logits in enumerate(intermediate_logits):
        weight = decay_factor ** (num_outputs - 1 - i)
        loss = compute_loss_with_mask(logits, targets, pad_token_id)
        total_loss += weight * loss
    total_weight = sum(decay_factor ** i for i in range(num_outputs))
    return total_loss / total_weight


# =============================================================================
# CONTROL TRANSFORMER TRAINING LOOP
# =============================================================================

def train_control_transformer(
    model, train_dataset, val_dataset, input_tokenizer, ans_tokenizer, epochs,
    learning_rate=1e-4, checkpoint_dir='checkpoints/control', log_dir='logs/control'
):
    """
    Training loop for the control (single-pass) transformer.
    This serves as the baseline for comparison with the recursive model.
    """
    print("\n" + "="*60)
    print("TRAINING CONTROL TRANSFORMER (Single-Pass Baseline)")
    print("="*60 + "\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    
    # Setup WandB
    try:
        wandb.init(
            project="tinyai",
            name="control-transformer",
            config={
                "model": "control_transformer",
                "learning_rate": learning_rate,
                "epochs": epochs,
                "d_model": model.d_model,
                "num_layers": model.num_layers,
                "num_heads": model.num_heads,
            }
        )
        use_wandb = True
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Continuing without WandB logging...")
        use_wandb = False
    
    train_metrics = TrainingMetrics()
    val_metrics = TrainingMetrics()
    best_val_loss = float('inf')
    global_step = 0
    
    @tf.function
    def train_step(inputs, targets):
        """Single training step for control transformer."""
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = compute_loss_with_mask(logits, targets, ans_tokenizer.pad_token_id)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy = compute_accuracy(logits, targets, ans_tokenizer.pad_token_id)
        return loss, accuracy
    
    @tf.function
    def val_step(inputs, targets):
        """Validation step for control transformer."""
        logits = model(inputs, training=False)
        loss = compute_loss_with_mask(logits, targets, ans_tokenizer.pad_token_id)
        accuracy = compute_accuracy(logits, targets, ans_tokenizer.pad_token_id)
        return loss, accuracy
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_metrics.reset()
        
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            loss, accuracy = train_step(inputs, targets)
            train_metrics.update(loss, accuracy)
            global_step += 1
            
            # Log to WandB
            if use_wandb:
                wandb.log({
                    'train/loss': float(loss),
                    'train/accuracy': float(accuracy),
                    'train/step': global_step
                })
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        # Validation
        val_metrics.reset()
        for inputs, targets in val_dataset:
            loss, accuracy = val_step(inputs, targets)
            val_metrics.update(loss, accuracy)
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'val/loss': float(val_metrics.avg_loss),
                'val/accuracy': float(val_metrics.avg_accuracy),
                'epoch': epoch + 1
            })
        
        print(f"\n  Train Loss: {train_metrics.avg_loss:.4f}, Train Acc: {train_metrics.avg_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics.avg_loss:.4f}, Val Acc: {val_metrics.avg_accuracy:.4f}")
        
        # Save best model
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.weights.h5'))
    
    # Save final model
    model.save_weights(os.path.join(checkpoint_dir, 'final_model.weights.h5'))
    print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f}")
    
    # Finish WandB run
    if use_wandb:
        wandb.finish()
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': train_metrics.avg_loss,
        'final_train_acc': train_metrics.avg_accuracy,
        'final_val_loss': val_metrics.avg_loss,
        'final_val_acc': val_metrics.avg_accuracy
    }


# =============================================================================
# RECURSIVE TRANSFORMER TRAINING LOOP
# =============================================================================

def train_recursive_transformer(
    model, train_dataset, val_dataset, input_tokenizer, ans_tokenizer, epochs,
    learning_rate=1e-4, checkpoint_dir='checkpoints/recursive', log_dir='logs/recursive'
):
    """
    Training loop for the recursive (TRM-inspired) transformer.
    
    The RecursiveTransformer has its own train_step method that implements:
    - Deep supervision across multiple recursion cycles
    - ACT (Adaptive Computation Time) loss for learning when to halt
    - Latent state management between recursion cycles
    
    This training loop wraps the model's built-in train_step for compatibility
    with our training infrastructure while preserving the TRM training dynamics.
    """
    print("\n" + "="*60)
    print("TRAINING RECURSIVE TRANSFORMER (TRM-Inspired)")
    print("="*60)
    print(f"  Deep recursion cycles (T): {model.deep_rec_cycles}")
    print(f"  Inner iterations (n): {model.num_l_steps}")
    print(f"  Deep supervision steps: {model.deep_sup_steps}")
    print(f"  ACT loss weight: {model.act_loss_weight}")
    print(f"  Halt exploration prob: {model.halt_exploration_prob}")
    print(f"  Max halt steps: {model.halt_max_steps}")
    print("="*60 + "\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup optimizer
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    model.compile(optimizer=optimizer)
    
    # Setup WandB
    try:
        wandb.init(
            project="tinyai",
            name="recursive-transformer",
            config={
                "model": "recursive_transformer",
                "learning_rate": learning_rate,
                "epochs": epochs,
                "d_model": model.d_model,
                "num_layers": model.num_layers,
                "num_heads": model.num_heads,
                "deep_rec_cycles": model.deep_rec_cycles,
                "num_l_steps": model.num_l_steps,
                "deep_sup_steps": model.deep_sup_steps,
                "act_loss_weight": model.act_loss_weight,
            }
        )
        use_wandb = True
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Continuing without WandB logging...")
        use_wandb = False
    
    train_metrics = TrainingMetrics()
    val_metrics = TrainingMetrics()
    best_val_loss = float('inf')
    global_step = 0
    
    @tf.function
    def val_step(inputs, targets):
        """Validation step for recursive transformer."""
        logits = model(inputs, training=False)
        loss = compute_loss_with_mask(logits, targets, ans_tokenizer.pad_token_id)
        accuracy = compute_accuracy(logits, targets, ans_tokenizer.pad_token_id)
        return loss, accuracy
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_metrics.reset()
        
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            # Use model's custom train_step which handles:
            # - Deep supervision across recursion cycles
            # - ACT loss for halting
            # - Latent state management
            result = model.train_step((inputs, targets))
            loss = result['loss']
            
            # Compute accuracy separately for logging
            logits = model(inputs, training=False)
            accuracy = compute_accuracy(logits, targets, ans_tokenizer.pad_token_id)
            
            train_metrics.update(loss, accuracy)
            global_step += 1
            
            # Log to WandB
            if use_wandb:
                wandb.log({
                    'train/loss': float(loss),
                    'train/accuracy': float(accuracy),
                    'train/step': global_step
                })
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        # Validation
        val_metrics.reset()
        for inputs, targets in val_dataset:
            loss, accuracy = val_step(inputs, targets)
            val_metrics.update(loss, accuracy)
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                'val/loss': float(val_metrics.avg_loss),
                'val/accuracy': float(val_metrics.avg_accuracy),
                'epoch': epoch + 1
            })
        
        print(f"\n  Train Loss: {train_metrics.avg_loss:.4f}, Train Acc: {train_metrics.avg_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics.avg_loss:.4f}, Val Acc: {val_metrics.avg_accuracy:.4f}")
        
        # Save best model
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.weights.h5'))
    
    # Save final model
    model.save_weights(os.path.join(checkpoint_dir, 'final_model.weights.h5'))
    print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f}")
    
    # Finish WandB run
    if use_wandb:
        wandb.finish()
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': train_metrics.avg_loss,
        'final_train_acc': train_metrics.avg_accuracy,
        'final_val_loss': val_metrics.avg_loss,
        'final_val_acc': val_metrics.avg_accuracy
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, test_dataset, ans_tokenizer, model_name="Model"):
    """Evaluate model on test set."""
    print(f"\nEvaluating {model_name}...")
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for inputs, targets in test_dataset:
        logits = model(inputs, training=False)
        loss = compute_loss_with_mask(logits, targets, ans_tokenizer.pad_token_id)
        accuracy = compute_accuracy(logits, targets, ans_tokenizer.pad_token_id)
        
        total_loss += float(loss)
        total_accuracy += float(accuracy)
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Test Accuracy: {avg_accuracy:.4f}")
    
    return {'test_loss': avg_loss, 'test_accuracy': avg_accuracy}


def compare_models(control_results, recursive_results):
    """Compare control and recursive model results."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Control':<15} {'Recursive':<15} {'Improvement':<15}")
    print("-"*70)
    
    for metric in ['test_loss', 'test_accuracy']:
        control_val = control_results.get(metric, 0)
        recursive_val = recursive_results.get(metric, 0)
        
        if 'loss' in metric:
            improvement = ((control_val - recursive_val) / control_val) * 100 if control_val > 0 else 0
            better = "↓" if recursive_val < control_val else "↑"
        else:
            improvement = ((recursive_val - control_val) / control_val) * 100 if control_val > 0 else 0
            better = "↑" if recursive_val > control_val else "↓"
        
        print(f"{metric:<25} {control_val:<15.4f} {recursive_val:<15.4f} {better} {abs(improvement):.2f}%")
    
    print("="*60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TinyAI Training Script')
    
    # Data arguments
    parser.add_argument('--full_data_path', type=str, default='preprocessing/data/final_train_data.txt')
    parser.add_argument('--q_data_path', type=str, default='preprocessing/data/questions.txt')
    parser.add_argument('--a_data_path', type=str, default='preprocessing/data/answers.txt')
    parser.add_argument('--input_vocab_path', type=str, default='preprocessing/data/input_vocab.json',
                        help='Path to input vocabulary file (input_vocab.json)')
    parser.add_argument('--ans_vocab_path', type=str, default='preprocessing/data/ans_vocab.json',
                        help='Path to output vocabulary file (ans_vocab.json)')
    parser.add_argument('--max_seq_length', type=int, default=256)
    
    # Model architecture arguments
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # Recursive transformer specific arguments
    parser.add_argument('--deep_rec_cycles', type=int, default=3,
                        help='Number of deep recursion cycles (T)')
    parser.add_argument('--num_l_steps', type=int, default=6,
                        help='Number of inner iterations (n)')
    parser.add_argument('--deep_sup_steps', type=int, default=4,
                        help='Number of deep supervision steps')
    parser.add_argument('--act_loss_weight', type=float, default=0.1,
                        help='Weight for ACT (halting) loss')
    parser.add_argument('--step_penalty_weight', type=float, default=0.01,
                        help='Weight for step penalty to encourage more refinement steps')
    parser.add_argument('--halt_exploration_prob', type=float, default=0.1,
                        help='Threshold for halting during inference')
    parser.add_argument('--halt_max_steps', type=int, default=16,
                        help='Maximum steps before forced halt')
    parser.add_argument('--min_halt_steps', type=int, default=3,
                        help='Minimum steps before halting is allowed')
    parser.add_argument('--stage_weights', type=str, default='',
                        help='Comma-separated weights for deep supervision stages (e.g., "0.3,0.7,1.0"); empty for uniform')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    # Control flags
    parser.add_argument('--skip_control', action='store_true',
                        help='Skip training control transformer')
    parser.add_argument('--skip_recursive', action='store_true',
                        help='Skip training recursive transformer')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(script_dir, args.full_data_path)
    q_data_path = os.path.join(script_dir, args.q_data_path)
    a_data_path = os.path.join(script_dir, args.a_data_path)
    input_vocab_path = os.path.join(script_dir, args.input_vocab_path)
    ans_vocab_path = os.path.join(script_dir, args.ans_vocab_path)
    
    # Print configuration
    print("\n" + "="*60)
    print("TinyAI TRAINING SCRIPT")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Full Data path: {full_data_path}")
    print(f"  Question Data path: {q_data_path}")
    print(f"  Answer Data path: {a_data_path}")
    print(f"  Input Vocab path: {input_vocab_path}")
    print(f"  Output Vocab path: {ans_vocab_path}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Heads: {args.num_heads}")
    print(f"  FF dim: {args.ff_dim}")
    print(f"  Dropout: {args.dropout_rate}")
    print(f"  Deep recursion cycles: {args.deep_rec_cycles}")
    print(f"  Inner iterations: {args.num_l_steps}")
    print(f"  Deep supervision steps: {args.deep_sup_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print("="*60 + "\n")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    input_tokenizer = Tokenizer(input_vocab_path)
    ans_tokenizer = Tokenizer(ans_vocab_path)
    
    # Load data
    print("Loading data...")
    data = load_data(q_data_path, a_data_path, input_tokenizer, ans_tokenizer, max_seq_length=args.max_seq_length)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    q_train, a_train = data['train']
    q_val, a_val = data['val']
    q_test, a_test = data['test']

    train_dataset = create_tf_dataset(q_train, a_train, args.batch_size, shuffle=True)
    val_dataset = create_tf_dataset(q_val, a_val, args.batch_size, shuffle=False)
    test_dataset = create_tf_dataset(q_test, a_test, args.batch_size, shuffle=False)

    results = {}
    
    # =========================================================================
    # CONTROL TRANSFORMER TRAINING LOOP
    # =========================================================================
    if not args.skip_control:
        print("\n" + "#"*60)
        print("# CONTROL TRANSFORMER")
        print("#"*60)
        
        control_model = Transformer(
            input_vocab_size=input_tokenizer.vocab_size,
            ans_vocab_size=ans_tokenizer.vocab_size,
            d_model=args.d_model,
            max_seq_length=args.max_seq_length,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout_rate=args.dropout_rate
        )
        
        # Build model
        dummy_input = tf.zeros((1, args.max_seq_length - 1), dtype=tf.int32)
        _ = control_model(dummy_input)
        print(f"\nControl Model Parameters: {control_model.count_params():,}")
        
        control_train_results = train_control_transformer(
            model=control_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_tokenizer=input_tokenizer,
            ans_tokenizer=ans_tokenizer,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints/control'),
            log_dir=os.path.join(args.output_dir, 'logs/control')
        )
        
        control_test_results = evaluate_model(control_model, test_dataset, ans_tokenizer, "Control Transformer")
        results['control'] = {**control_train_results, **control_test_results}
    
    # =========================================================================
    # RECURSIVE TRANSFORMER TRAINING LOOP
    # =========================================================================
    if not args.skip_recursive:
        print("\n" + "#"*60)
        print("# RECURSIVE TRANSFORMER (TRM-Inspired)")
        print("#"*60)
        
        recursive_model = RecursiveTransformer(
            input_vocab_size=input_tokenizer.vocab_size,
            ans_vocab_size=ans_tokenizer.vocab_size,
            d_model=args.d_model,
            max_seq_length=args.max_seq_length,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout_rate=args.dropout_rate,
            deep_rec_cycles=args.deep_rec_cycles,
            deep_sup_steps=args.deep_sup_steps,
            act_loss_weight=args.act_loss_weight,
            step_penalty_weight=args.step_penalty_weight,
            num_l_steps=args.num_l_steps,
            halt_exploration_prob=args.halt_exploration_prob,
            halt_max_steps=args.halt_max_steps,
            min_halt_steps=args.min_halt_steps,
            stage_weights=[float(x) for x in args.stage_weights.split(",")] if args.stage_weights else None
        )
        
        # Build model
        dummy_input = tf.zeros((1, args.max_seq_length - 1), dtype=tf.int32)
        _ = recursive_model(dummy_input)
        print(f"\nRecursive Model Parameters: {recursive_model.count_params():,}")
        
        recursive_train_results = train_recursive_transformer(
            model=recursive_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            input_tokenizer=input_tokenizer,
            ans_tokenizer=ans_tokenizer,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints/recursive'),
            log_dir=os.path.join(args.output_dir, 'logs/recursive')
        )
        
        recursive_test_results = evaluate_model(recursive_model, test_dataset, ans_tokenizer, "Recursive Transformer")
        results['recursive'] = {**recursive_train_results, **recursive_test_results}
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    if 'control' in results and 'recursive' in results:
        compare_models(results['control'], results['recursive'])
    
    # Save results
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

