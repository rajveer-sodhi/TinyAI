#!/usr/bin/env python3
"""
WandB Login Script for OSCAR

Run this script once to authenticate with WandB.
After logging in, you can monitor training progress at https://wandb.ai

Usage:
    python wandb_login.py
"""

import wandb

print("=" * 60)
print("WandB Login for TinyAI")
print("=" * 60)
print("")
print("This will open a browser window or show you a link to authenticate.")
print("After logging in, your training runs will automatically upload metrics to WandB.")
print("You can view them at: https://wandb.ai")
print("")
print("Press Enter to continue...")
input()

wandb.login()

print("")
print("=" * 60)
print("Login complete!")
print("=" * 60)
print("")
print("You can now submit training jobs and monitor them at:")
print("  https://wandb.ai/your-username/tinyai")
print("")
print("To view metrics during training, just visit the WandB dashboard.")
print("No port forwarding needed!")

