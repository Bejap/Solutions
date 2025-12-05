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
from collections import deque
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
    DQN_AGENT_POSITIONS,
    REPLAY_MEMORY_SIZE,
    USE_PRIORITIZED_REPLAY,
    PER_ALPHA,
    PER_BETA_START,
    PER_BETA_FRAMES,
    PER_EPSILON
)
from Whist.utils.prioritized_replay import PrioritizedReplayMemory


def load_embedded_agent_from_weights(weights_path: str, embedding_dim: int, 
                                    gamma: float, agent_id: int = 0,
                                    use_double_dqn: bool = True,
                                    use_prioritized_replay: bool = None) -> EmbeddedDQNAgent:
    """
    Load an Embedded DQN agent from saved weights.
    
    Args:
        weights_path: Path to the saved weights file (.h5 or .weights.h5)
        embedding_dim: Dimension of card embeddings
        gamma: Discount factor
        agent_id: ID for the agent
        use_double_dqn: Whether to use Double DQN
        use_prioritized_replay: Whether to use prioritized replay (None = use global setting)
        
    Returns:
        Loaded EmbeddedDQNAgent instance
    """
    print(f"Loading embedded agent from weights: {weights_path}")
    
    # Use global setting if not specified
    if use_prioritized_replay is None:
        use_prioritized_replay = USE_PRIORITIZED_REPLAY
    
    # Create new agent with same architecture
    agent = EmbeddedDQNAgent(
        embedding_dim=embedding_dim, 
        gamma=gamma, 
        agent_id=agent_id,
        use_double_dqn=use_double_dqn,
        use_prioritized_replay=use_prioritized_replay
    )
    
    # Load weights
    agent.model.load_weights(weights_path)
    agent.target_model.set_weights(agent.model.get_weights())
    
    print(f"Embedded agent {agent_id} loaded successfully with Double DQN={use_double_dqn}, PER={use_prioritized_replay}")
    return agent


def load_embedded_agent_from_keras(model_path: str, gamma: float, 
                                   embedding_dim: int = 8, agent_id: int = 0,
                                   use_double_dqn: bool = True,
                                   use_prioritized_replay: bool = None) -> EmbeddedDQNAgent:
    """
    Load an Embedded DQN agent from a saved Keras model.
    
    IMPORTANT: This creates a minimal agent wrapper around the loaded model.
    The loaded model's architecture is used as-is without modification.
    
    Args:
        model_path: Path to the saved Keras model (.keras)
        gamma: Discount factor
        embedding_dim: Dimension of card embeddings (not used when loading)
        agent_id: ID for the agent
        use_double_dqn: Whether to use Double DQN
        use_prioritized_replay: Whether to use prioritized replay (None = use global setting)
        
    Returns:
        Loaded EmbeddedDQNAgent instance
    """
    print(f"Loading full embedded agent model: {model_path}")
    
    # Use global setting if not specified
    if use_prioritized_replay is None:
        use_prioritized_replay = USE_PRIORITIZED_REPLAY
    
    # Load the Keras model without compiling (avoids custom object issues)
    # compile=False skips loading optimizer state which often causes problems
    try:
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        print("  Model loaded successfully (without compilation)")
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("  Trying with custom objects...")
        # If that fails, try with custom_objects parameter
        try:
            loaded_model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
            )
        except Exception as e2:
            print(f"  Error with custom objects: {e2}")
            print("  Trying with safe_mode...")
            # Last resort: use safe mode
            loaded_model = tf.keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False
            )
    
    # Create a basic agent structure - but we'll use the loaded model directly
    # We need to bypass the normal __init__ to avoid creating a new model
    agent = object.__new__(EmbeddedDQNAgent)
    agent.agent_id = agent_id
    agent.embedding_dim = embedding_dim
    agent.use_double_dqn = use_double_dqn
    agent.use_prioritized_replay = use_prioritized_replay
    agent.gamma = gamma
    
    # Set the loaded model
    agent.model = loaded_model
    
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
    
    # Initialize replay memory based on setting
    if use_prioritized_replay:
        agent.replay_memory = PrioritizedReplayMemory(
            capacity=REPLAY_MEMORY_SIZE,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES,
            epsilon=PER_EPSILON
        )
        print(f"  Using prioritized experience replay")
    else:
        agent.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        print(f"  Using standard replay memory")
    
    agent.target_update_counter = 0
    
    print(f"Embedded agent {agent_id} loaded successfully from Keras model")
    print(f"  Model has {len(loaded_model.inputs) if isinstance(loaded_model.input, list) else 1} inputs")
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
                 use_prioritized_replay: bool = None,
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
            use_prioritized_replay: Whether to use prioritized replay (None = use global setting)
                                   Set to False to load old models without prioritized replay
            enable_per_card_reward: Enable per-card rewards
        """
        self.starting_episode = starting_episode
        self.num_additional_games = num_additional_games
        self.path_type = path_type
        self.embedding_dim = embedding_dim
        
        # Convert paths to absolute paths if they're relative
        if not os.path.isabs(agent_0_path):
            agent_0_path = os.path.join(project_root, agent_0_path)
        if not os.path.isabs(agent_2_path):
            agent_2_path = os.path.join(project_root, agent_2_path)
        
        # Verify files exist before trying to load
        if path_type == 'weights':
            if not os.path.exists(agent_0_path):
                raise FileNotFoundError(f"Agent 0 weights file not found: {agent_0_path}")
            if not os.path.exists(agent_2_path):
                raise FileNotFoundError(f"Agent 2 weights file not found: {agent_2_path}")
        elif path_type == 'keras':
            if not os.path.exists(agent_0_path):
                raise FileNotFoundError(f"Agent 0 keras file not found: {agent_0_path}")
            if not os.path.exists(agent_2_path):
                raise FileNotFoundError(f"Agent 2 keras file not found: {agent_2_path}")
        
        # Load agents
        print("="*60)
        print("LOADING PRE-TRAINED EMBEDDED AGENTS")
        print("="*60)
        print(f"Agent 0 path: {agent_0_path}")
        print(f"Agent 2 path: {agent_2_path}")
        print(f"Files exist: Agent 0={os.path.exists(agent_0_path)}, Agent 2={os.path.exists(agent_2_path)}")
        print("="*60)
        
        gamma_values = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
        
        if path_type == 'weights':
            self.agent_0 = load_embedded_agent_from_weights(
                agent_0_path, embedding_dim, gamma_values[0], 
                agent_id=0, use_double_dqn=use_double_dqn,
                use_prioritized_replay=use_prioritized_replay
            )
            self.agent_2 = load_embedded_agent_from_weights(
                agent_2_path, embedding_dim, gamma_values[2],
                agent_id=2, use_double_dqn=use_double_dqn,
                use_prioritized_replay=use_prioritized_replay
            )
        elif path_type == 'keras':
            self.agent_0 = load_embedded_agent_from_keras(
                agent_0_path, gamma_values[0], embedding_dim,
                agent_id=0, use_double_dqn=use_double_dqn,
                use_prioritized_replay=use_prioritized_replay
            )
            self.agent_2 = load_embedded_agent_from_keras(
                agent_2_path, gamma_values[2], embedding_dim,
                agent_id=2, use_double_dqn=use_double_dqn,
                use_prioritized_replay=use_prioritized_replay
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
    # IMPORTANT: Backward Compatibility for Old Models
    # ================================================
    # If you're loading models trained BEFORE prioritized replay was added,
    # set use_prioritized_replay=False to match the old architecture.
    # 
    # For NEW models (trained with prioritized replay), you can:
    # - Set use_prioritized_replay=True (or None to use global setting)
    # - Or omit it to use the global USE_PRIORITIZED_REPLAY setting
    
    # IMPORTANT: File Paths
    # ====================
    # Paths can be:
    # 1. Relative to project root (recommended): "Weights/embedded/model.h5"
    #    - Will be automatically converted to absolute path
    # 2. Absolute path: "C:/full/path/to/Weights/embedded/model.h5"
    #    - Works directly as-is
    # 
    # The script will verify files exist before attempting to load them
    
    # Example: Load from weights and continue training
    # Update these paths to your actual model files
    agent_0_weights = "Weights/embedded/embedded_agent_player_0_ep1000_avgR-3.25_20251201_120000.weights.h5"
    agent_2_weights = "Weights/embedded/embedded_agent_player_2_ep1000_avgR-3.25_20251201_120000.weights.h5"
    
    # Or load from Keras models
    # agent_0_model = "Models/embedded/embedded_full_agent_player_0_ep1000_avgR-3.25_20251201_120000.keras"
    # agent_2_model = "Models/embedded/embedded_full_agent_player_2_ep1000_avgR-3.25_20251201_120000.keras"
    
    # Continue training
    # For OLD models (trained before PER was added): set use_prioritized_replay=False
    # For NEW models (trained with PER): set use_prioritized_replay=True or omit
    continue_trainer = ContinueEmbeddedTraining(
        agent_0_path=agent_0_weights,
        agent_2_path=agent_2_weights,
        path_type='weights',  # Use 'keras' for full models
        embedding_dim=8,
        starting_episode=1000,
        num_additional_games=2000,
        epsilon=0.3,  # Lower epsilon since agents are already trained
        use_double_dqn=True,
        use_prioritized_replay=False  # Set to False for old models, True/None for new models
    )
    
    # Run training
    trainer = continue_trainer.continue_training()
    
    # Plot results
    trainer.plot_results()


if __name__ == "__main__":
    print("="*60)
    print("CONTINUE EMBEDDED TRAINING SCRIPT")
    print("="*60)
    print("\nIMPORTANT NOTES:")
    print("1. Update the model paths in main() to match your saved models")
    print("2. Models must be compatible with current EmbeddedDQNAgent architecture")
    print("3. Current architecture uses card embeddings")
    print("4. Prioritized replay is now enabled by default")
    print("="*60)
    print()
    
    main()
