"""
Model Training Script for DQN Agent

This script trains the Whist DQN agent using one-hot encoding.

Usage:
    python model_training.py
    
    OR from project root:
    python -m Whist.training.model_training
"""

# Add project root to path for direct script execution
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Whist.training.training_logic import WhistTrainer
from Whist.utils.constants import (
    DEFAULT_NUM_GAMES,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    ARRAY_LENGTH,
    DEFAULT_GAMMA_VALUES,
    DEFAULT_SAVE_EVERY
)

if __name__ == "__main__":
    # Create and configure the trainer
    trainer = WhistTrainer(
        num_games=DEFAULT_NUM_GAMES,
        epsilon=DEFAULT_EPSILON,
        epsilon_decay=DEFAULT_EPSILON_DECAY,
        min_epsilon=DEFAULT_MIN_EPSILON,
        array_length=ARRAY_LENGTH,
        gamma_values=DEFAULT_GAMMA_VALUES,
        save_every=DEFAULT_SAVE_EVERY
    )
    
    # Run the training
    trainer.train()
    
    # Plot the results
    trainer.plot_results()
