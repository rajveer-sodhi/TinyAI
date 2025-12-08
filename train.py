"""
TinyAI Training Script

This script implements training loops for:
1. Control Transformer: Standard single-pass transformer baseline
2. Recursive Transformer: TRM-inspired recursive refinement transformer

Based on the paper "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from collections import Counter

# Add components directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))

from transformer import Transformer
from embeddings import EmbeddingLayer
from transformer_encoder import TransformerEncoder


# =============================================================================
# RECURSIVE TRANSFORMER MODEL
# =============================================================================

class RecursiveTransformer(keras.Model):
    """
    Recursive Transformer implementing TRM-inspired iterative refinement.
    
    Key concepts from the TRM paper:
    - y: Current solution embedding (refined across iterations)
    - z: Latent reasoning state (hidden state that persists across iterations)
    - T: Number of deep recursion cycles
    - n: Number of inner iterations per cycle
    - Deep supervision: Loss computed at multiple refinement steps
    """
    
    def __init__(
        self,
        vocab_size,
        d_model,
        max_seq_length,
        num_layers,
        num_heads,
        ff_dim,
        num_recursions=3,       # T: deep recursion cycles
        inner_iterations=6,     # n: inner iterations per cycle
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_recursions = num_recursions
        self.inner_iterations = inner_iterations
        self.dropout_rate = dropout_rate
        
        # Embedding layer for input tokens
        self.embedding = EmbeddingLayer(vocab_size, d_model, max_seq_length, dropout_rate)
        
        # Transformer encoder (shared across all recursion steps)
        self.transformer_encoder = TransformerEncoder(d_model, num_heads, ff_dim, num_layers, dropout_rate)
        
        # Latent state projection layers
        self.z_projection = keras.layers.Dense(d_model, activation='gelu', name='z_projection')
        self.y_update = keras.layers.Dense(d_model, name='y_update')
        
        # Combination layer for merging input, current solution, and latent state
        self.combine_layer = keras.layers.Dense(d_model, name='combine')
        
        # Final layers
        self.layernorm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.output_head = keras.layers.Dense(vocab_size)
    
    def init_states(self, batch_size, seq_length):
        """Initialize y (solution) and z (latent reasoning state)."""
        y = tf.zeros((batch_size, seq_length, self.d_model))
        z = tf.zeros((batch_size, seq_length, self.d_model))
        return y, z
    
    def single_iteration(self, x_embed, y, z, training=False):
        """
        Perform a single recursion iteration.
        
        Args:
            x_embed: Embedded input (batch, seq, d_model)
            y: Current solution state (batch, seq, d_model)
            z: Latent reasoning state (batch, seq, d_model)
            training: Whether in training mode
            
        Returns:
            Updated y and z states
        """
        # Combine input embedding, current solution, and latent state
        combined = self.combine_layer(tf.concat([x_embed, y, z], axis=-1))
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(combined, training=training)
        
        # Update latent state z
        z_new = self.z_projection(encoded)
        
        # Update solution y using the new latent state
        y_new = y + self.y_update(z_new)
        
        return y_new, z_new
    
    def call(self, inputs, training=False, return_intermediates=False):
        """
        Forward pass with recursive refinement.
        
        Args:
            inputs: Input token IDs (batch, seq)
            training: Whether in training mode
            return_intermediates: If True, return all intermediate predictions
            
        Returns:
            Final logits (or list of intermediate logits if return_intermediates=True)
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Get input embeddings
        x_embed = self.embedding(inputs, training=training)
        
        # Initialize states
        y, z = self.init_states(batch_size, seq_length)
        
        intermediate_outputs = []
        
        # Deep recursion cycles
        for t in range(self.num_recursions):
            # Inner iterations
            for n in range(self.inner_iterations):
                # First T-1 cycles: no gradient (refinement without backprop)
                # Last cycle: gradient enabled for training
                if training and t < self.num_recursions - 1:
                    y_detached = tf.stop_gradient(y)
                    z_detached = tf.stop_gradient(z)
                    y, z = self.single_iteration(x_embed, y_detached, z_detached, training=training)
                else:
                    y, z = self.single_iteration(x_embed, y, z, training=training)
                
                # Compute intermediate output for deep supervision
                if return_intermediates:
                    output = self.layernorm(y)
                    output = self.dropout(output, training=training)
                    logits = self.output_head(output)
                    intermediate_outputs.append(logits)
        
        # Final output
        output = self.layernorm(y)
        output = self.dropout(output, training=training)
        logits = self.output_head(output)
        
        if return_intermediates:
            intermediate_outputs.append(logits)
            return intermediate_outputs
        
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_length': self.max_seq_length,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_recursions': self.num_recursions,
            'inner_iterations': self.inner_iterations,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# =============================================================================
# DATA LOADING AND TOKENIZATION
# =============================================================================

class SimpleTokenizer:
    """Character-level tokenizer for math word problems."""
    
    def __init__(self, max_vocab_size=5000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
    def fit(self, texts):
        """Build vocabulary from texts."""
        # Count character frequencies
        char_counts = Counter()
        for text in texts:
            # Handle special tokens by replacing them temporarily
            text_clean = text.replace('[BOS]', ' ').replace('[EOS]', ' ')
            for char in text_clean:
                char_counts[char] += 1
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # Add most common characters
        idx = len(self.special_tokens)
        for char, _ in char_counts.most_common(self.max_vocab_size - len(self.special_tokens)):
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1
        
        print(f"Vocabulary size: {len(self.token_to_id)}")
        return self
    
    @property
    def vocab_size(self):
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self):
        return self.token_to_id[self.pad_token]
    
    def encode(self, text, max_length=None, padding=True):
        """Convert text to token IDs."""
        tokens = []
        
        # Check for special tokens
        i = 0
        while i < len(text):
            found_special = False
            for special in self.special_tokens:
                if text[i:].startswith(special):
                    tokens.append(self.token_to_id[special])
                    i += len(special)
                    found_special = True
                    break
            
            if not found_special:
                char = text[i]
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                else:
                    tokens.append(self.token_to_id[self.unk_token])
                i += 1
        
        # Truncate or pad
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif padding and len(tokens) < max_length:
                tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        
        return tokens
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        chars = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if token != self.pad_token:
                    chars.append(token)
        return ''.join(chars)
    
    def save(self, path):
        """Save tokenizer to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'max_vocab_size': self.max_vocab_size
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(max_vocab_size=data['max_vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(v): k for k, v in data['token_to_id'].items()}
        return tokenizer


def load_data(data_path, tokenizer=None, max_seq_length=512, test_split=0.1, val_split=0.1):
    """
    Load and preprocess training data.
    
    Args:
        data_path: Path to the training data file
        tokenizer: Optional pre-fitted tokenizer
        max_seq_length: Maximum sequence length
        test_split: Fraction for test set
        val_split: Fraction for validation set
        
    Returns:
        Dictionary with train/val/test datasets and tokenizer
    """
    # Load raw data
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(lines)} samples")
    
    # Fit tokenizer if not provided
    if tokenizer is None:
        tokenizer = SimpleTokenizer()
        tokenizer.fit(lines)
    
    # Tokenize all data
    encoded_data = []
    for line in lines:
        tokens = tokenizer.encode(line, max_length=max_seq_length, padding=True)
        encoded_data.append(tokens)
    
    encoded_data = np.array(encoded_data, dtype=np.int32)
    
    # Shuffle data
    np.random.seed(42)
    indices = np.random.permutation(len(encoded_data))
    encoded_data = encoded_data[indices]
    
    # Split data
    n_total = len(encoded_data)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_test - n_val
    
    train_data = encoded_data[:n_train]
    val_data = encoded_data[n_train:n_train + n_val]
    test_data = encoded_data[n_train + n_val:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'tokenizer': tokenizer
    }


def create_tf_dataset(data, batch_size, shuffle=True):
    """
    Create TensorFlow dataset for language modeling.
    
    For language modeling, input is tokens[:-1] and target is tokens[1:]
    """
    inputs = data[:, :-1]
    targets = data[:, 1:]
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

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
    """Compute token-level accuracy, ignoring padding."""
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    
    # Create mask for non-padding tokens
    mask = tf.cast(targets != pad_token_id, tf.float32)
    
    # Compute correct predictions
    correct = tf.cast(predictions == targets, tf.float32)
    correct = correct * mask
    
    # Compute accuracy
    accuracy = tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)
    
    return accuracy


def compute_loss_with_mask(logits, targets, pad_token_id):
    """Compute cross-entropy loss, ignoring padding tokens."""
    # Create mask
    mask = tf.cast(targets != pad_token_id, tf.float32)
    
    # Compute loss
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        targets, logits, from_logits=True
    )
    
    # Apply mask
    loss = loss * mask
    
    # Average over non-padding tokens
    loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)
    
    return loss


def compute_deep_supervision_loss(intermediate_logits, targets, pad_token_id, decay_factor=0.8):
    """
    Compute deep supervision loss across all intermediate outputs.
    
    Earlier outputs contribute less (weighted by decay_factor).
    """
    total_loss = 0.0
    num_outputs = len(intermediate_logits)
    
    for i, logits in enumerate(intermediate_logits):
        # Weight increases for later outputs
        weight = decay_factor ** (num_outputs - 1 - i)
        loss = compute_loss_with_mask(logits, targets, pad_token_id)
        total_loss += weight * loss
    
    # Normalize by total weight
    total_weight = sum(decay_factor ** i for i in range(num_outputs))
    total_loss = total_loss / total_weight
    
    return total_loss


# =============================================================================
# TRAINING LOOPS
# =============================================================================

def train_control_transformer(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    epochs,
    learning_rate=1e-4,
    checkpoint_dir='checkpoints/control',
    log_dir='logs/control'
):
    """
    Training loop for the control (single-pass) transformer.
    
    Args:
        model: Transformer model instance
        train_dataset: Training TF dataset
        val_dataset: Validation TF dataset
        tokenizer: Tokenizer instance
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """
    print("\n" + "="*60)
    print("TRAINING CONTROL TRANSFORMER (Single-Pass Baseline)")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.01
    )
    
    # TensorBoard writer
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))
    
    # Metrics trackers
    train_metrics = TrainingMetrics()
    val_metrics = TrainingMetrics()
    
    # Best validation loss for checkpointing
    best_val_loss = float('inf')
    global_step = 0
    
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = compute_loss_with_mask(logits, targets, tokenizer.pad_token_id)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        accuracy = compute_accuracy(logits, targets, tokenizer.pad_token_id)
        
        return loss, accuracy
    
    @tf.function
    def val_step(inputs, targets):
        logits = model(inputs, training=False)
        loss = compute_loss_with_mask(logits, targets, tokenizer.pad_token_id)
        accuracy = compute_accuracy(logits, targets, tokenizer.pad_token_id)
        return loss, accuracy
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_metrics.reset()
        
        # Training
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            loss, accuracy = train_step(inputs, targets)
            train_metrics.update(loss, accuracy)
            
            global_step += 1
            
            # Log to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=global_step)
                tf.summary.scalar('accuracy', accuracy, step=global_step)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        # Validation
        val_metrics.reset()
        for inputs, targets in val_dataset:
            loss, accuracy = val_step(inputs, targets)
            val_metrics.update(loss, accuracy)
        
        # Log validation metrics
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_metrics.avg_loss, step=epoch)
            tf.summary.scalar('accuracy', val_metrics.avg_accuracy, step=epoch)
        
        print(f"\n  Train Loss: {train_metrics.avg_loss:.4f}, Train Acc: {train_metrics.avg_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics.avg_loss:.4f}, Val Acc: {val_metrics.avg_accuracy:.4f}")
        
        # Save checkpoint if best validation loss
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.weights.h5'))
    
    # Save final model
    model.save_weights(os.path.join(checkpoint_dir, 'final_model.weights.h5'))
    print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': train_metrics.avg_loss,
        'final_train_acc': train_metrics.avg_accuracy,
        'final_val_loss': val_metrics.avg_loss,
        'final_val_acc': val_metrics.avg_accuracy
    }


def train_recursive_transformer(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    epochs,
    learning_rate=1e-4,
    use_deep_supervision=True,
    supervision_decay=0.8,
    checkpoint_dir='checkpoints/recursive',
    log_dir='logs/recursive'
):
    """
    Training loop for the recursive transformer with TRM-inspired refinement.
    
    Args:
        model: RecursiveTransformer model instance
        train_dataset: Training TF dataset
        val_dataset: Validation TF dataset
        tokenizer: Tokenizer instance
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        use_deep_supervision: Whether to use deep supervision loss
        supervision_decay: Decay factor for deep supervision weights
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """
    print("\n" + "="*60)
    print("TRAINING RECURSIVE TRANSFORMER (TRM-Inspired)")
    print("="*60)
    print(f"  Recursion cycles (T): {model.num_recursions}")
    print(f"  Inner iterations (n): {model.inner_iterations}")
    print(f"  Deep supervision: {use_deep_supervision}")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=0.01
    )
    
    # TensorBoard writer
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))
    
    # Metrics trackers
    train_metrics = TrainingMetrics()
    val_metrics = TrainingMetrics()
    
    # Best validation loss for checkpointing
    best_val_loss = float('inf')
    global_step = 0
    
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            if use_deep_supervision:
                # Get all intermediate outputs for deep supervision
                intermediate_logits = model(inputs, training=True, return_intermediates=True)
                loss = compute_deep_supervision_loss(
                    intermediate_logits, targets, tokenizer.pad_token_id, supervision_decay
                )
                logits = intermediate_logits[-1]  # Use final output for accuracy
            else:
                logits = model(inputs, training=True, return_intermediates=False)
                loss = compute_loss_with_mask(logits, targets, tokenizer.pad_token_id)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Gradient clipping (important for recursive models)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        accuracy = compute_accuracy(logits, targets, tokenizer.pad_token_id)
        
        return loss, accuracy, grad_norm
    
    @tf.function
    def val_step(inputs, targets):
        logits = model(inputs, training=False, return_intermediates=False)
        loss = compute_loss_with_mask(logits, targets, tokenizer.pad_token_id)
        accuracy = compute_accuracy(logits, targets, tokenizer.pad_token_id)
        return loss, accuracy
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_metrics.reset()
        
        # Training
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            loss, accuracy, grad_norm = train_step(inputs, targets)
            train_metrics.update(loss, accuracy)
            
            global_step += 1
            
            # Log to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=global_step)
                tf.summary.scalar('accuracy', accuracy, step=global_step)
                tf.summary.scalar('gradient_norm', grad_norm, step=global_step)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Loss={loss:.4f}, Acc={accuracy:.4f}, GradNorm={grad_norm:.4f}")
        
        # Validation
        val_metrics.reset()
        for inputs, targets in val_dataset:
            loss, accuracy = val_step(inputs, targets)
            val_metrics.update(loss, accuracy)
        
        # Log validation metrics
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_metrics.avg_loss, step=epoch)
            tf.summary.scalar('accuracy', val_metrics.avg_accuracy, step=epoch)
        
        print(f"\n  Train Loss: {train_metrics.avg_loss:.4f}, Train Acc: {train_metrics.avg_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics.avg_loss:.4f}, Val Acc: {val_metrics.avg_accuracy:.4f}")
        
        # Save checkpoint if best validation loss
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.weights.h5'))
    
    # Save final model
    model.save_weights(os.path.join(checkpoint_dir, 'final_model.weights.h5'))
    print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f}")
    
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

def evaluate_model(model, test_dataset, tokenizer, model_name="Model"):
    """Evaluate model on test set."""
    print(f"\nEvaluating {model_name}...")
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for inputs, targets in test_dataset:
        logits = model(inputs, training=False)
        loss = compute_loss_with_mask(logits, targets, tokenizer.pad_token_id)
        accuracy = compute_accuracy(logits, targets, tokenizer.pad_token_id)
        
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
    parser.add_argument('--data_path', type=str, 
                       default='preprocessing/data/final_train_data.txt',
                       help='Path to training data')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    
    # Recursive model arguments
    parser.add_argument('--num_recursions', type=int, default=3,
                       help='Number of deep recursion cycles (T)')
    parser.add_argument('--inner_iterations', type=int, default=6,
                       help='Number of inner iterations per cycle (n)')
    parser.add_argument('--use_deep_supervision', action='store_true', default=True,
                       help='Use deep supervision loss')
    parser.add_argument('--supervision_decay', type=float, default=0.8,
                       help='Decay factor for deep supervision')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    
    # Mode arguments
    parser.add_argument('--train_control', action='store_true', default=True,
                       help='Train control transformer')
    parser.add_argument('--train_recursive', action='store_true', default=True,
                       help='Train recursive transformer')
    parser.add_argument('--skip_control', action='store_true',
                       help='Skip control transformer training')
    parser.add_argument('--skip_recursive', action='store_true',
                       help='Skip recursive transformer training')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Resolve data path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data_path)
    
    print("\n" + "="*60)
    print("TinyAI TRAINING SCRIPT")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Heads: {args.num_heads}")
    print(f"  FF dim: {args.ff_dim}")
    print(f"  Recursion cycles: {args.num_recursions}")
    print(f"  Inner iterations: {args.inner_iterations}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data...")
    data = load_data(
        data_path,
        max_seq_length=args.max_seq_length
    )
    
    tokenizer = data['tokenizer']
    
    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))
    
    # Create datasets
    train_dataset = create_tf_dataset(data['train'], args.batch_size, shuffle=True)
    val_dataset = create_tf_dataset(data['val'], args.batch_size, shuffle=False)
    test_dataset = create_tf_dataset(data['test'], args.batch_size, shuffle=False)
    
    results = {}
    
    # =========================================================================
    # CONTROL TRANSFORMER TRAINING LOOP
    # =========================================================================
    if args.train_control and not args.skip_control:
        print("\n" + "#"*60)
        print("# CONTROL TRANSFORMER")
        print("#"*60)
        
        # Create control model
        control_model = Transformer(
            vocab_size=tokenizer.vocab_size,
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
        
        # Train
        control_train_results = train_control_transformer(
            model=control_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints/control'),
            log_dir=os.path.join(args.output_dir, 'logs/control')
        )
        
        # Evaluate
        control_test_results = evaluate_model(
            control_model, test_dataset, tokenizer, "Control Transformer"
        )
        
        results['control'] = {**control_train_results, **control_test_results}
    
    # =========================================================================
    # RECURSIVE TRANSFORMER TRAINING LOOP
    # =========================================================================
    if args.train_recursive and not args.skip_recursive:
        print("\n" + "#"*60)
        print("# RECURSIVE TRANSFORMER (TRM-Inspired)")
        print("#"*60)
        
        # Create recursive model
        recursive_model = RecursiveTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            max_seq_length=args.max_seq_length,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_recursions=args.num_recursions,
            inner_iterations=args.inner_iterations,
            dropout_rate=args.dropout_rate
        )
        
        # Build model
        dummy_input = tf.zeros((1, args.max_seq_length - 1), dtype=tf.int32)
        _ = recursive_model(dummy_input)
        
        print(f"\nRecursive Model Parameters: {recursive_model.count_params():,}")
        
        # Train
        recursive_train_results = train_recursive_transformer(
            model=recursive_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            use_deep_supervision=args.use_deep_supervision,
            supervision_decay=args.supervision_decay,
            checkpoint_dir=os.path.join(args.output_dir, 'checkpoints/recursive'),
            log_dir=os.path.join(args.output_dir, 'logs/recursive')
        )
        
        # Evaluate
        recursive_test_results = evaluate_model(
            recursive_model, test_dataset, tokenizer, "Recursive Transformer"
        )
        
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

