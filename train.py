"""
TinyAI Training Script

Training loops for control transformer and recursive transformer.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'components'))

from transformer import Transformer

# TODO: Import RecursiveTransformer once implemented
# from recursive_transformer import RecursiveTransformer

# TODO: Import Tokenizer once implemented
# from tokenizer import Tokenizer


def load_data(data_path, tokenizer, max_seq_length=512, test_split=0.1, val_split=0.1):
    with open(data_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(lines)} samples")
    
    encoded_data = []
    for line in lines:
        tokens = tokenizer.encode(line, max_length=max_seq_length, padding=True)
        encoded_data.append(tokens)
    
    encoded_data = np.array(encoded_data, dtype=np.int32)
    
    np.random.seed(42)
    indices = np.random.permutation(len(encoded_data))
    encoded_data = encoded_data[indices]
    
    n_total = len(encoded_data)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_test - n_val
    
    train_data = encoded_data[:n_train]
    val_data = encoded_data[n_train:n_train + n_val]
    test_data = encoded_data[n_train + n_val:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return {'train': train_data, 'val': val_data, 'test': test_data, 'tokenizer': tokenizer}


def create_tf_dataset(data, batch_size, shuffle=True):
    inputs = data[:, :-1]
    targets = data[:, 1:]
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


class TrainingMetrics:
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
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    mask = tf.cast(targets != pad_token_id, tf.float32)
    correct = tf.cast(predictions == targets, tf.float32) * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)


def compute_loss_with_mask(logits, targets, pad_token_id):
    mask = tf.cast(targets != pad_token_id, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def compute_deep_supervision_loss(intermediate_logits, targets, pad_token_id, decay_factor=0.8):
    total_loss = 0.0
    num_outputs = len(intermediate_logits)
    for i, logits in enumerate(intermediate_logits):
        weight = decay_factor ** (num_outputs - 1 - i)
        loss = compute_loss_with_mask(logits, targets, pad_token_id)
        total_loss += weight * loss
    total_weight = sum(decay_factor ** i for i in range(num_outputs))
    return total_loss / total_weight


def train_control_transformer(
    model, train_dataset, val_dataset, tokenizer, epochs,
    learning_rate=1e-4, checkpoint_dir='checkpoints/control', log_dir='logs/control'
):
    print("\n" + "="*60)
    print("TRAINING CONTROL TRANSFORMER (Single-Pass Baseline)")
    print("="*60 + "\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))
    
    train_metrics = TrainingMetrics()
    val_metrics = TrainingMetrics()
    best_val_loss = float('inf')
    global_step = 0
    
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = compute_loss_with_mask(logits, targets, tokenizer.pad_token_id)
        
        gradients = tape.gradient(loss, model.trainable_variables)
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
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_metrics.reset()
        
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            loss, accuracy = train_step(inputs, targets)
            train_metrics.update(loss, accuracy)
            global_step += 1
            
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=global_step)
                tf.summary.scalar('accuracy', accuracy, step=global_step)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        val_metrics.reset()
        for inputs, targets in val_dataset:
            loss, accuracy = val_step(inputs, targets)
            val_metrics.update(loss, accuracy)
        
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_metrics.avg_loss, step=epoch)
            tf.summary.scalar('accuracy', val_metrics.avg_accuracy, step=epoch)
        
        print(f"\n  Train Loss: {train_metrics.avg_loss:.4f}, Train Acc: {train_metrics.avg_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics.avg_loss:.4f}, Val Acc: {val_metrics.avg_accuracy:.4f}")
        
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.weights.h5'))
    
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
    model, train_dataset, val_dataset, tokenizer, epochs,
    learning_rate=1e-4, use_deep_supervision=True, supervision_decay=0.8,
    checkpoint_dir='checkpoints/recursive', log_dir='logs/recursive'
):
    print("\n" + "="*60)
    print("TRAINING RECURSIVE TRANSFORMER (TRM-Inspired)")
    print("="*60)
    print(f"  Recursion cycles (T): {model.num_recursions}")
    print(f"  Inner iterations (n): {model.inner_iterations}")
    print(f"  Deep supervision: {use_deep_supervision}")
    print("="*60 + "\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))
    
    train_metrics = TrainingMetrics()
    val_metrics = TrainingMetrics()
    best_val_loss = float('inf')
    global_step = 0
    
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            if use_deep_supervision:
                intermediate_logits = model(inputs, training=True, return_intermediates=True)
                loss = compute_deep_supervision_loss(
                    intermediate_logits, targets, tokenizer.pad_token_id, supervision_decay
                )
                logits = intermediate_logits[-1]
            else:
                logits = model(inputs, training=True, return_intermediates=False)
                loss = compute_loss_with_mask(logits, targets, tokenizer.pad_token_id)
        
        gradients = tape.gradient(loss, model.trainable_variables)
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
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        train_metrics.reset()
        
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            loss, accuracy, grad_norm = train_step(inputs, targets)
            train_metrics.update(loss, accuracy)
            global_step += 1
            
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=global_step)
                tf.summary.scalar('accuracy', accuracy, step=global_step)
                tf.summary.scalar('gradient_norm', grad_norm, step=global_step)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: Loss={loss:.4f}, Acc={accuracy:.4f}, GradNorm={grad_norm:.4f}")
        
        val_metrics.reset()
        for inputs, targets in val_dataset:
            loss, accuracy = val_step(inputs, targets)
            val_metrics.update(loss, accuracy)
        
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_metrics.avg_loss, step=epoch)
            tf.summary.scalar('accuracy', val_metrics.avg_accuracy, step=epoch)
        
        print(f"\n  Train Loss: {train_metrics.avg_loss:.4f}, Train Acc: {train_metrics.avg_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics.avg_loss:.4f}, Val Acc: {val_metrics.avg_accuracy:.4f}")
        
        if val_metrics.avg_loss < best_val_loss:
            best_val_loss = val_metrics.avg_loss
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        if (epoch + 1) % 5 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.weights.h5'))
    
    model.save_weights(os.path.join(checkpoint_dir, 'final_model.weights.h5'))
    print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': train_metrics.avg_loss,
        'final_train_acc': train_metrics.avg_accuracy,
        'final_val_loss': val_metrics.avg_loss,
        'final_val_acc': val_metrics.avg_accuracy
    }


def evaluate_model(model, test_dataset, tokenizer, model_name="Model"):
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


def main():
    parser = argparse.ArgumentParser(description='TinyAI Training Script')
    
    parser.add_argument('--data_path', type=str, default='preprocessing/data/final_train_data.txt')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--num_recursions', type=int, default=3)
    parser.add_argument('--inner_iterations', type=int, default=6)
    parser.add_argument('--use_deep_supervision', action='store_true', default=True)
    parser.add_argument('--supervision_decay', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--skip_control', action='store_true')
    parser.add_argument('--skip_recursive', action='store_true')
    parser.add_argument('--output_dir', type=str, default='output')
    
    args = parser.parse_args()
    
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
    
    # TODO: Initialize tokenizer once implemented
    # tokenizer = Tokenizer()
    # tokenizer.fit(data_path)
    raise NotImplementedError("Tokenizer not implemented. Need tokenizer with: vocab_size, pad_token_id, encode(text, max_length, padding), decode(ids)")
    
    print("Loading data...")
    data = load_data(data_path, tokenizer, max_seq_length=args.max_seq_length)
    tokenizer = data['tokenizer']
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_dataset = create_tf_dataset(data['train'], args.batch_size, shuffle=True)
    val_dataset = create_tf_dataset(data['val'], args.batch_size, shuffle=False)
    test_dataset = create_tf_dataset(data['test'], args.batch_size, shuffle=False)
    
    results = {}
    
    # CONTROL TRANSFORMER TRAINING LOOP
    if not args.skip_control:
        print("\n" + "#"*60)
        print("# CONTROL TRANSFORMER")
        print("#"*60)
        
        control_model = Transformer(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            max_seq_length=args.max_seq_length,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout_rate=args.dropout_rate
        )
        
        dummy_input = tf.zeros((1, args.max_seq_length - 1), dtype=tf.int32)
        _ = control_model(dummy_input)
        print(f"\nControl Model Parameters: {control_model.count_params():,}")
        
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
        
        control_test_results = evaluate_model(control_model, test_dataset, tokenizer, "Control Transformer")
        results['control'] = {**control_train_results, **control_test_results}
    
    # RECURSIVE TRANSFORMER TRAINING LOOP
    if not args.skip_recursive:
        print("\n" + "#"*60)
        print("# RECURSIVE TRANSFORMER (TRM-Inspired)")
        print("#"*60)
        
        # TODO: Create recursive model once implemented
        # recursive_model = RecursiveTransformer(
        #     vocab_size=tokenizer.vocab_size,
        #     d_model=args.d_model,
        #     max_seq_length=args.max_seq_length,
        #     num_layers=args.num_layers,
        #     num_heads=args.num_heads,
        #     ff_dim=args.ff_dim,
        #     num_recursions=args.num_recursions,
        #     inner_iterations=args.inner_iterations,
        #     dropout_rate=args.dropout_rate
        # )
        raise NotImplementedError("RecursiveTransformer not implemented. Need model with: num_recursions, inner_iterations, call(inputs, training, return_intermediates)")
        
        dummy_input = tf.zeros((1, args.max_seq_length - 1), dtype=tf.int32)
        _ = recursive_model(dummy_input)
        print(f"\nRecursive Model Parameters: {recursive_model.count_params():,}")
        
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
        
        recursive_test_results = evaluate_model(recursive_model, test_dataset, tokenizer, "Recursive Transformer")
        results['recursive'] = {**recursive_train_results, **recursive_test_results}
    
    if 'control' in results and 'recursive' in results:
        compare_models(results['control'], results['recursive'])
    
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
