"""
Loading and Continue Training for Embedded DQN Agent

This module provides functionality to load pre-trained Embedded DQN agents
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
from Whist.core.whist_embedded import WhistEmbedded
from Whist.agents.embedded_dqn_agent import EmbeddedDQNAgent
from Whist.agents.ew_strategy import EWStrategy
from Whist.training.embedded.training_embedded import EmbeddedWhistTrainer
from Whist.utils.constants import (
    DEFAULT_NUM_GAMES,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    CARDS_PER_PLAYER,
    DEFAULT_GAMMA_VALUES,
    NUM_PLAYERS,
    DQN_AGENT_POSITIONS
)


def load_embedded_agent_from_weights(weights_path: str, embedding_dim: int, 
                                    gamma: float, agent_id: int = 0,
                                    use_double_dqn: bool = True) -> EmbeddedDQNAgent:
    """
    Load an Embedded DQN agent from saved weights.
    
    Args:
        weights_path: Path to the saved weights file (.h5 or .weights.h5)
        embedding_dim: Dimension of card embeddings
        gamma: Discount factor
        agent_id: ID for the agent
        use_double_dqn: Whether to use Double DQN
        
    Returns:
        Loaded EmbeddedDQNAgent instance
    """
    print(f"Loading embedded agent from weights: {weights_path}")
    
    # Create new agent with same architecture
    agent = EmbeddedDQNAgent(
        embedding_dim=embedding_dim, 
        gamma=gamma, 
        agent_id=agent_id,
        use_double_dqn=use_double_dqn
    )
    
    # Load weights
    agent.model.load_weights(weights_path)
    agent.target_model.set_weights(agent.model.get_weights())
    
    print(f"Embedded agent {agent_id} loaded successfully with Double DQN={use_double_dqn}")
    return agent


def load_embedded_agent_from_keras(model_path: str, gamma: float, 
                                   embedding_dim: int = 8, agent_id: int = 0,
                                   use_double_dqn: bool = True) -> EmbeddedDQNAgent:
    """
    Load an Embedded DQN agent from a saved Keras model.
    
    Args:
        model_path: Path to the saved Keras model (.keras)
        gamma: Discount factor
        embedding_dim: Dimension of card embeddings
        agent_id: ID for the agent
        use_double_dqn: Whether to use Double DQN
        
    Returns:
        Loaded EmbeddedDQNAgent instance
    """
    print(f"Loading full embedded agent model: {model_path}")
    
    # Load the Keras model
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Create agent and assign loaded model
    agent = EmbeddedDQNAgent(
        embedding_dim=embedding_dim,
        gamma=gamma, 
        agent_id=agent_id,
        use_double_dqn=use_double_dqn
    )
    
    # Replace model with loaded one
    agent.model = loaded_model
    agent.target_model.set_weights(agent.model.get_weights())
    
    print(f"Embedded agent {agent_id} loaded successfully from Keras model")
    return agent


class ContinueEmbeddedTraining:
    """
    Class for loading embedded agents and continuing training from checkpoint.
    """
    
    def __init__(self, agent_0_path: str, agent_2_path: str,
                 path_type: str = 'weights',
                 embedding_dim: int = 8,
                 starting_episode: int = 0,
                 num_additional_games: int = DEFAULT_NUM_GAMES,
                 epsilon: float = DEFAULT_EPSILON,
                 epsilon_decay: float = DEFAULT_EPSILON_DECAY,
                 min_epsilon: float = DEFAULT_MIN_EPSILON,
                 gamma_values: list = None,
                 use_double_dqn: bool = True,
                 enable_per_card_reward: bool = True):
        """
        Initialize continue training setup for embedded agents.
        
        Args:
            agent_0_path: Path to agent 0 checkpoint
            agent_2_path: Path to agent 2 checkpoint
            path_type: Type of checkpoint ('weights' or 'keras')
            embedding_dim: Dimension of card embeddings
            starting_episode: Episode number to start from
            num_additional_games: Number of additional games to train
            epsilon: Starting exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon
            gamma_values: Discount factors for agents
            use_double_dqn: Whether to use Double DQN
            enable_per_card_reward: Enable per-card rewards
        """
        self.starting_episode = starting_episode
        self.num_additional_games = num_additional_games
        self.path_type = path_type
        self.embedding_dim = embedding_dim
        
        # Load agents
        print("="*60)
        print("LOADING PRE-TRAINED EMBEDDED AGENTS")
        print("="*60)
        
        gamma_values = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
        
        if path_type == 'weights':
            self.agent_0 = load_embedded_agent_from_weights(
                agent_0_path, embedding_dim, gamma_values[0], 
                agent_id=0, use_double_dqn=use_double_dqn
            )
            self.agent_2 = load_embedded_agent_from_weights(
                agent_2_path, embedding_dim, gamma_values[2],
                agent_id=2, use_double_dqn=use_double_dqn
            )
        elif path_type == 'keras':
            self.agent_0 = load_embedded_agent_from_keras(
                agent_0_path, gamma_values[0], embedding_dim,
                agent_id=0, use_double_dqn=use_double_dqn
            )
            self.agent_2 = load_embedded_agent_from_keras(
                agent_2_path, gamma_values[2], embedding_dim,
                agent_id=2, use_double_dqn=use_double_dqn
            )
        else:
            raise ValueError(f"Unknown path_type: {path_type}. Use 'weights' or 'keras'")
        
        # Create trainer
        self.trainer = EmbeddedWhistTrainer(
            embedding_dim=embedding_dim,
            num_games=num_additional_games,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            gamma_values=gamma_values,
            enable_per_card_reward=enable_per_card_reward
        )
        
        # Replace trainer's agents with loaded ones
        self.trainer.agents[0] = self.agent_0
        self.trainer.agents[2] = self.agent_2
        
        print(f"\nEmbedded agents loaded (embedding_dim={embedding_dim})")
        print(f"Ready to continue training from episode {starting_episode}")
        print(f"Will train for {num_additional_games} additional episodes")
        print(f"Starting epsilon: {epsilon:.4f}")
        print("="*60)
    
    def continue_training(self):
        """Continue training from checkpoint."""
        print("\nContinuing embedded training...")
        self.trainer.train()
        print("\nTraining completed!")
        
        return self.trainer
    
    def get_agents(self):
        """Get the loaded agents."""
        return self.agent_0, self.agent_2


def main():
    """Example usage of continue embedded training."""
    # Example: Load from weights and continue training
    agent_0_weights = "Weights/embedded_agent_player_0_ep1000_avgR-3.25.weights.h5"
    agent_2_weights = "Weights/embedded_agent_player_2_ep1000_avgR-3.25.weights.h5"
    
    # Or load from Keras models
    # agent_0_model = "Models/embedded_full_agent_player_0_ep1000_avgR-3.25.keras"
    # agent_2_model = "Models/embedded_full_agent_player_2_ep1000_avgR-3.25.keras"
    
    # Continue training
    continue_trainer = ContinueEmbeddedTraining(
        agent_0_path=agent_0_weights,
        agent_2_path=agent_2_weights,
        path_type='weights',
        embedding_dim=8,
        starting_episode=1000,
        num_additional_games=2000,
        epsilon=0.3,  # Lower epsilon since agents are already trained
        use_double_dqn=True
    )
    
    # Run training
    trainer = continue_trainer.continue_training()
    
    # Plot results
    trainer.plot_results()


if __name__ == "__main__":
    main()
