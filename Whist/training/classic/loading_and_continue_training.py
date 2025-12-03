"""
Loading and Continue Training for Standard DQN Agent

This module provides functionality to load pre-trained DQN agents
and continue training from a checkpoint.
"""

# Add project root to Python path to allow running script directly
import sys
import os
from pathlib import Path

# Get the project root (3 levels up from this file)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
from collections import deque
from Whist.core.whist import Whist
from Whist.agents.simple_whist_DQN import DQNAgent
from Whist.agents.ew_strategy import EWStrategy
from Whist.training.classic.training_logic import WhistTrainer
from Whist.utils.constants import (
    DEFAULT_NUM_GAMES,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    ARRAY_LENGTH,
    DEFAULT_GAMMA_VALUES,
    NUM_PLAYERS,
    DQN_AGENT_POSITIONS,
    EAST,
    WEST,
    REPLAY_MEMORY_SIZE,
    USE_PRIORITIZED_REPLAY,
    PER_ALPHA,
    PER_BETA_START,
    PER_BETA_FRAMES,
    PER_EPSILON
)
from Whist.utils.prioritized_replay import PrioritizedReplayMemory


def load_agent_from_weights(weights_path: str, input_size: int, gamma: float, 
                            agent_id: int = 0, use_double_dqn: bool = True) -> DQNAgent:
    """
    Load a DQN agent from saved weights.
    
    Args:
        weights_path: Path to the saved weights file (.h5 or .weights.h5)
        input_size: Size of the input layer
        gamma: Discount factor
        agent_id: ID for the agent
        use_double_dqn: Whether to use Double DQN
        
    Returns:
        Loaded DQNAgent instance
    """
    print(f"Loading agent from weights: {weights_path}")
    
    # Create new agent with same architecture
    agent = DQNAgent(input_size, gamma, agent_id=agent_id, use_double_dqn=use_double_dqn)
    
    # Load weights
    agent.model.load_weights(weights_path)
    agent.target_model.set_weights(agent.model.get_weights())
    
    print(f"Agent {agent_id} loaded successfully with Double DQN={use_double_dqn}")
    return agent


def load_agent_from_keras(model_path: str, gamma: float, agent_id: int = 0, 
                          use_double_dqn: bool = True) -> DQNAgent:
    """
    Load a DQN agent from a saved Keras model.
    
    IMPORTANT: This creates a minimal agent wrapper around the loaded model.
    The loaded model's architecture is used as-is without modification.
    
    Args:
        model_path: Path to the saved Keras model (.keras)
        gamma: Discount factor
        agent_id: ID for the agent
        use_double_dqn: Whether to use Double DQN
        
    Returns:
        Loaded DQNAgent instance with the model
    """
    print(f"Loading full agent model: {model_path}")
    
    # Load the Keras model without compiling (avoids custom object issues)
    # compile=False skips loading optimizer state which often causes problems
    try:
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        print("  Model loaded successfully (without compilation)")
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Trying with custom objects...")
        # If that fails, try with custom_objects parameter
        loaded_model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
    
    # Extract input size from the loaded model
    # For multi-input models, sum all input dimensions
    if isinstance(loaded_model.input_shape, list):
        # Multiple inputs
        input_size = sum(shape[1] if shape[1] is not None else 0 for shape in loaded_model.input_shape)
    else:
        # Single input
        input_size = loaded_model.input_shape[1]
    
    print(f"  Detected input size from model: {input_size}")
    
    # Create a basic agent structure - but we'll use the loaded model directly
    # We need to bypass the normal __init__ to avoid creating a new model
    agent = object.__new__(DQNAgent)
    agent.agent_id = agent_id
    agent.input_shape = input_size
    agent.use_double_dqn = use_double_dqn
    agent.use_prioritized_replay = USE_PRIORITIZED_REPLAY
    agent.gamma = gamma
    
    # Set the loaded model
    agent.model = loaded_model
    agent.critic = None  # Not used when loading from Keras
    
    # Recompile the loaded model with fresh optimizer
    agent.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    # Create target model as a copy of the loaded model
    agent.target_model = tf.keras.models.clone_model(loaded_model)
    agent.target_model.set_weights(agent.model.get_weights())
    agent.target_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError()
    )
    agent.target_critic = None  # Not used when loading from Keras
    
    # Initialize replay memory
    if USE_PRIORITIZED_REPLAY:
        agent.replay_memory = PrioritizedReplayMemory(
            capacity=REPLAY_MEMORY_SIZE,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES,
            epsilon=PER_EPSILON
        )
    else:
        agent.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    
    agent.target_update_counter = 0
    
    print(f"Agent {agent_id} loaded successfully from Keras model")
    print(f"  Model architecture: {len(loaded_model.inputs) if isinstance(loaded_model.input, list) else 1} inputs, {len(loaded_model.outputs)} outputs")
    return agent


class ContinueTraining:
    """
    Class for loading agents and continuing training from checkpoint.
    """
    
    def __init__(self, agent_0_path: str, agent_2_path: str, 
                 path_type: str = 'weights',
                 starting_episode: int = 0,
                 num_additional_games: int = DEFAULT_NUM_GAMES,
                 epsilon: float = DEFAULT_EPSILON,
                 epsilon_decay: float = DEFAULT_EPSILON_DECAY,
                 min_epsilon: float = DEFAULT_MIN_EPSILON,
                 gamma_values: list = None,
                 use_double_dqn: bool = True,
                 enable_per_card_reward: bool = True,
                 early_stopping_patience: int = 100):
        """
        Initialize continue training setup.
        
        Args:
            agent_0_path: Path to agent 0 checkpoint
            agent_2_path: Path to agent 2 checkpoint
            path_type: Type of checkpoint ('weights' or 'keras')
            starting_episode: Episode number to start from
            num_additional_games: Number of additional games to train
            epsilon: Starting exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon
            gamma_values: Discount factors for agents
            use_double_dqn: Whether to use Double DQN
            enable_per_card_reward: Enable per-card rewards
            early_stopping_patience: Patience for early stopping
        """
        self.starting_episode = starting_episode
        self.num_additional_games = num_additional_games
        self.path_type = path_type
        
        # Load agents
        print("="*60)
        print("LOADING PRE-TRAINED AGENTS")
        print("="*60)
        
        input_size = (ARRAY_LENGTH * 7) + 4 + 4
        gamma_values = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
        
        if path_type == 'weights':
            self.agent_0 = load_agent_from_weights(
                agent_0_path, input_size, gamma_values[0], agent_id=0, 
                use_double_dqn=use_double_dqn
            )
            self.agent_2 = load_agent_from_weights(
                agent_2_path, input_size, gamma_values[2], agent_id=2,
                use_double_dqn=use_double_dqn
            )
        elif path_type == 'keras':
            self.agent_0 = load_agent_from_keras(
                agent_0_path, gamma_values[0], agent_id=0,
                use_double_dqn=use_double_dqn
            )
            self.agent_2 = load_agent_from_keras(
                agent_2_path, gamma_values[2], agent_id=2,
                use_double_dqn=use_double_dqn
            )
        else:
            raise ValueError(f"Unknown path_type: {path_type}. Use 'weights' or 'keras'")
        
        # Create trainer
        self.trainer = WhistTrainer(
            num_games=num_additional_games,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            gamma_values=gamma_values,
            enable_per_card_reward=enable_per_card_reward,
            early_stopping_patience=early_stopping_patience
        )
        
        # Replace trainer's agents with loaded ones
        self.trainer.agents[0] = self.agent_0
        self.trainer.agents[2] = self.agent_2
        
        print(f"\nAgents loaded. Ready to continue training from episode {starting_episode}")
        print(f"Will train for {num_additional_games} additional episodes")
        print(f"Starting epsilon: {epsilon:.4f}")
        print("="*60)
    
    def continue_training(self):
        """Continue training from checkpoint."""
        print("\nContinuing training...")
        self.trainer.train()
        print("\nTraining completed!")
        
        return self.trainer
    
    def get_agents(self):
        """Get the loaded agents."""
        return self.agent_0, self.agent_2


def main():
    """Example usage of continue training."""
    # IMPORTANT: Make sure your saved models are compatible with the current architecture
    # The current DQNAgent uses multiple inputs (game, player, tracking, score)
    
    # Example: Load from weights and continue training
    # Update these paths to your actual model files
    agent_0_weights = "Weights/dueling/agent_player_0_ep600_avgR-2.73_20251203_120008.weights.h5"
    agent_2_weights = "Weights/dueling/agent_player_2_ep600_avgR-2.73_20251203_120008.weights.h5"
    
    # Or load from Keras models
    # agent_0_model = "Models/dueling/full_agent_player_0_ep600_avgR-2.73_20251203_120008.keras"
    # agent_2_model = "Models/dueling/full_agent_player_2_ep600_avgR-2.73_20251203_120008.keras"
    
    # Continue training
    continue_trainer = ContinueTraining(
        agent_0_path=agent_0_weights,
        agent_2_path=agent_2_weights,
        path_type='weights',  # Use 'keras' for full models
        starting_episode=500,
        num_additional_games=2000,
        epsilon=0.9,
        use_double_dqn=True,
        early_stopping_patience=150
    )
    
    # Run training
    trainer = continue_trainer.continue_training()
    
    # Plot results
    trainer.plot_results()


if __name__ == "__main__":
    print("="*60)
    print("CONTINUE TRAINING SCRIPT")
    print("="*60)
    print("\nIMPORTANT NOTES:")
    print("1. Update the model paths in main() to match your saved models")
    print("2. Models must be compatible with current DQNAgent architecture")
    print("3. Current architecture uses multi-input (game/player/tracking/score)")
    print("4. Prioritized replay is now enabled by default")
    print("="*60)
    print()
    
    main()
