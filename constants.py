"""
Constants for the Whist DQN Training Project

This module contains all constants used across the project, including:
- Game configuration
- Training hyperparameters
- DQN agent parameters
- Model architecture parameters
"""

# =============================================================================
# GAME CONFIGURATION
# =============================================================================

# Number of cards in the game (Hearts 2-A)
ARRAY_LENGTH = 13

# Number of players
NUM_PLAYERS = 4

# Player positions (teams: 0-2 and 1-3 are partners)
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Team configuration (which positions are DQN agents vs EW strategy)
DQN_AGENT_POSITIONS = [NORTH, SOUTH]  # Positions 0 and 2
EW_STRATEGY_POSITIONS = [EAST, WEST]  # Positions 1 and 3


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Default training configuration
DEFAULT_NUM_GAMES = 1000
DEFAULT_EPSILON = 1.0  # Initial exploration rate
DEFAULT_EPSILON_DECAY = 0.996
DEFAULT_MIN_EPSILON = 0.001

# Gamma values for different agents (discount factor for future rewards)
DEFAULT_GAMMA_VALUES = [0.99, 0.95, 0.90, 0.85]

# How often to save models during training
DEFAULT_SAVE_EVERY = 500

# Legacy training parameters (from whist.py)
LEGACY_EPISODES = 250
LEGACY_EPSILON_DECAY = 0.99


# =============================================================================
# DQN AGENT PARAMETERS
# =============================================================================

# Replay memory configuration
REPLAY_MEMORY_SIZE = 100  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum replay memory size before training starts
MINIBATCH_SIZE = 32  # Batch size for training (increased from 8 for stability)

# Target network update frequency
UPDATE_TARGET_EVERY = 5  # Update target network every N terminal states

# Model configuration
MODEL_NAME = 'smalle'
MIN_REWARD = -200  # Minimum reward threshold for model save
MEMORY_FRACTION = 0.35  # GPU memory fraction to use

# Default gamma value for DQN (if not using agent-specific values)
DEFAULT_GAMMA = 0.99


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# State size calculation: (ARRAY_LENGTH * 7) + 4 + 4 = 91 + 4 + 4 = 99
# Breaking down:
# - ARRAY_LENGTH * 2: cards_array + round_array = 26
# - ARRAY_LENGTH + 4: hand_array + player_array = 17
# - ARRAY_LENGTH * 4: tracking for 4 players = 52
# - 4: score_array
STATE_SIZE = 91  # Used in model_testing.py

# Action size (number of possible card actions)
ACTION_SIZE = 13  # Same as ARRAY_LENGTH

# Input dimensions for multi-input neural network
GAME_INPUT_SIZE = ARRAY_LENGTH * 2  # cards_array + round_array
PLAYER_INPUT_SIZE = ARRAY_LENGTH + 4  # hand_array + player_array
TRACKING_INPUT_SIZE = ARRAY_LENGTH * 4  # All 4 players' card tracking
SCORE_INPUT_SIZE = 4  # score_array for 4 players

# Network architecture
HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 64
HIDDEN_LAYER_3_SIZE = 32
DROPOUT_RATE = 0.35


# =============================================================================
# FILE PATHS AND SAVING
# =============================================================================

# Directories for saving models and weights
WEIGHTS_DIR = "Weights"
MODELS_DIR = "Models"

# File name patterns
AGENT_WEIGHTS_PATTERN = "{dir}/agent_player_{player}_ep{episode}.weights.h5"
FULL_AGENT_PATTERN = "{dir}/full_agent_player_{player}_ep{episode}.keras"


# =============================================================================
# STRATEGIC PLAY (EW Strategy)
# =============================================================================

# Probability of random play for EW strategy players
EW_RANDOM_PLAY_PROBABILITY = 0.2  # 20% random, 80% strategic
