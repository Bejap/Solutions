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

# Number of players
NUM_PLAYERS = 4

# Cards dealt to each player (adjust this for smaller games)
# For full whist game: 13 cards per player (52 total)
# For smaller games: 9 or 11 cards per player
CARDS_PER_PLAYER = 13

# Total number of cards in the game (calculated from cards per player)
# This represents the state array size for tracking all possible cards
# Note: Set CARDS_PER_PLAYER to adjust game size, not ARRAY_LENGTH directly
ARRAY_LENGTH = CARDS_PER_PLAYER * NUM_PLAYERS  # e.g., 13 * 4 = 52

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

# State size calculation: (ARRAY_LENGTH * 7) + 4 + 4
# Breaking down:
# - ARRAY_LENGTH * 2: cards_array + round_array
# - ARRAY_LENGTH + 4: hand_array + player_array
# - ARRAY_LENGTH * 4: tracking for 4 players
# - 4: score_array
# Total: (ARRAY_LENGTH * 7) + 8
# Example for 52 cards: (52 * 7) + 8 = 372
# Example for 44 cards (11 per player): (44 * 7) + 8 = 316
STATE_SIZE = (ARRAY_LENGTH * 7) + 8  # Automatically calculated based on ARRAY_LENGTH

# Action size (number of possible card actions) 
# Note: In practice, action space is dynamic based on hand size
ACTION_SIZE = CARDS_PER_PLAYER  # Cards per player

# Input dimensions for multi-input neural network
GAME_INPUT_SIZE = ARRAY_LENGTH * 2  # cards_array + round_array
PLAYER_INPUT_SIZE = ARRAY_LENGTH + 4  # hand_array + player_array
TRACKING_INPUT_SIZE = ARRAY_LENGTH * 4  # All 4 players' card tracking
SCORE_INPUT_SIZE = 4  # score_array for 4 players

# Network architecture
# Note: These can be adjusted based on game size
# Smaller games (9-11 cards) may benefit from smaller networks
HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 64
HIDDEN_LAYER_3_SIZE = 32
DROPOUT_RATE = 0.35

# For smaller games, you might want to use:
# HIDDEN_LAYER_1_SIZE = 64
# HIDDEN_LAYER_2_SIZE = 32
# HIDDEN_LAYER_3_SIZE = 16


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


# =============================================================================
# INHERITANCE & CLASS STRUCTURE
# =============================================================================

# Base class configuration
BASE_CLASSES_MODULE = "base_classes"

# Agent types for polymorphic behavior
AGENT_TYPE_DQN = "DQN"
AGENT_TYPE_STRATEGY = "STRATEGY"
AGENT_TYPE_RANDOM = "RANDOM"

# Strategy types
STRATEGY_EW = "EW_STRATEGY"  # East-West bridge-like strategy
STRATEGY_RANDOM = "RANDOM_STRATEGY"  # Random play strategy

# Player types
PLAYER_TYPE_HUMAN = "HUMAN"
PLAYER_TYPE_AI = "AI"
PLAYER_TYPE_STRATEGY = "STRATEGY"
