"""
Model Training Script for Embedded DQN Agent

This script trains the Whist DQN agent using card embeddings instead of
one-hot encoding. The embedded agent has a fixed state size regardless of
the number of cards per player, making it more flexible and compact.

Supports GPU/NPU acceleration for faster training.

Usage:
    python model_training_embedded.py
    
    OR from project root:
    python -m Whist.training.model_training_embedded
"""

# Add project root to path for direct script execution
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Whist.training.training_embedded import EmbeddedWhistTrainer
from Whist.utils.constants import (
    DEFAULT_NUM_GAMES,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    DEFAULT_GAMMA_VALUES,
    DEFAULT_SAVE_EVERY,
    CARDS_PER_PLAYER,
    USE_GPU
)
from Whist.utils.device_config import print_device_info

if __name__ == "__main__":
    print("=" * 70)
    print("WHIST DQN TRAINING - EMBEDDED AGENT")
    print("=" * 70)
    print()
    
    # Print device configuration
    print_device_info()
    
    print("Configuration:")
    print(f"  Embedding dimension: 8")
    print(f"  State size: 64 dimensions (fixed)")
    print(f"  Cards per player: {CARDS_PER_PLAYER}")
    print(f"  Number of episodes: {DEFAULT_NUM_GAMES}")
    print(f"  Initial epsilon: {DEFAULT_EPSILON}")
    print(f"  Epsilon decay: {DEFAULT_EPSILON_DECAY}")
    print(f"  Min epsilon: {DEFAULT_MIN_EPSILON}")
    print(f"  GPU Acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    print()
    print("Benefits of Embedded Agent:")
    print("  ✓ Fixed state size (works with 9, 11, or 13 cards)")
    print("  ✓ Compact representation (64 vs 372 dimensions)")
    print("  ✓ Learned card relationships")
    print("  ✓ Faster training")
    if USE_GPU:
        print("  ✓ GPU/NPU acceleration")
    print()
    print("-" * 70)
    print()
    
    # Create and configure the embedded trainer
    trainer = EmbeddedWhistTrainer(
        embedding_dim=8,
        num_games=DEFAULT_NUM_GAMES,
        epsilon=DEFAULT_EPSILON,
        epsilon_decay=DEFAULT_EPSILON_DECAY,
        min_epsilon=DEFAULT_MIN_EPSILON,
        gamma_values=DEFAULT_GAMMA_VALUES,
        save_every=DEFAULT_SAVE_EVERY
    )
    
    # Run the training
    trainer.train()
    
    # Plot the results
    trainer.plot_results()
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("Models saved to:")
    print("  - Weights/embedded_agent_player_*_avgR*_<timestamp>.weights.h5")
    print("  - Models/embedded_agent_player_*_avgR*_<timestamp>.keras")
    print()
    print("To change game size:")
    print("  1. Edit constants.py: CARDS_PER_PLAYER = 11")
    print("  2. Re-run this script")
    print("  Same embedded agent works for any card count!")

