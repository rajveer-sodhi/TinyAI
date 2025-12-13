#!/usr/bin/env python3
"""
Helper script to find available checkpoints and generate resume commands.
"""

import os
import json
from datetime import datetime

def find_checkpoints(checkpoint_dir='output/checkpoints/recursive'):
    """Find all available checkpoints."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.weights.h5'):
            filepath = os.path.join(checkpoint_dir, filename)
            modified_time = os.path.getmtime(filepath)
            size = os.path.getsize(filepath)
            
            # Extract epoch number if it's a periodic checkpoint
            epoch = None
            if 'epoch_' in filename:
                try:
                    epoch = int(filename.split('epoch_')[1].split('.')[0])
                except:
                    pass
            
            checkpoints.append({
                'filename': filename,
                'filepath': filepath,
                'epoch': epoch,
                'modified': datetime.fromtimestamp(modified_time),
                'size_mb': size / (1024 * 1024)
            })
    
    # Sort by epoch (if available) then by modified time
    checkpoints.sort(key=lambda x: (x['epoch'] if x['epoch'] is not None else -1, x['modified']), reverse=True)
    return checkpoints

def find_wandb_run_id():
    """Find the latest WandB run ID."""
    latest_run_file = "wandb/latest-run"
    if os.path.exists(latest_run_file):
        with open(latest_run_file, 'r') as f:
            latest_run = f.read().strip()
            # Extract run ID from "run-YYYYMMDD_HHMMSS-RUNID" format
            if '-' in latest_run:
                run_id = latest_run.split('-')[-1]
                return run_id, latest_run
    return None, None

def main():
    print("="*70)
    print("CHECKPOINT RESUME HELPER")
    print("="*70)
    
    # Find WandB run info
    print("\n📊 WandB Run Info:")
    run_id, run_dir = find_wandb_run_id()
    if run_id:
        print(f"  Latest run ID: {run_id}")
        print(f"  Run directory: {run_dir}")
        print(f"  Dashboard: https://wandb.ai/benjamin_peckham-brown-university/tinyai/runs/{run_id}")
    else:
        print("  No WandB run detected (will auto-detect when resuming)")
    
    # Find checkpoints
    print("\n💾 Available Checkpoints:")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("  No checkpoints found.")
        return
    
    for i, ckpt in enumerate(checkpoints, 1):
        epoch_str = f"Epoch {ckpt['epoch']}" if ckpt['epoch'] else "Best/Final"
        print(f"\n  [{i}] {ckpt['filename']}")
        print(f"      {epoch_str} | {ckpt['size_mb']:.1f} MB | {ckpt['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate resume commands
    print("\n" + "="*70)
    print("🚀 TO RESUME TRAINING:")
    print("="*70)
    
    # Show command for most recent checkpoint
    if checkpoints:
        latest = checkpoints[0]
        print("\n1. Edit train.sh and set these variables:")
        print(f"""
RESUME_CHECKPOINT="{latest['filepath']}"
RESUME_EPOCH={latest['epoch'] if latest['epoch'] else 0}
WANDB_RESUME_ID="{run_id if run_id else ''}"  # Optional - auto-detected if empty
""")
        
        print("2. Then run:")
        print("   sbatch train.sh --skip_control")
        
        print("\n" + "-"*70)
        print("OR pass arguments directly:")
        print("-"*70)
        
        cmd_parts = [
            f'--resume_from_checkpoint "{latest["filepath"]}"',
            f'--resume_epoch {latest["epoch"] if latest["epoch"] else 0}',
            '--skip_control'
        ]
        if run_id:
            cmd_parts.append(f'--wandb_resume_id "{run_id}"')
        
        print(f"\nsbatch train.sh {' '.join(cmd_parts)}")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()

