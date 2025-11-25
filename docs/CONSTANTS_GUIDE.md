# Constants.py Reference Guide

## Overview

The `constants.py` file serves as the centralized configuration hub for the entire Whist DQN project. All configurable parameters, hyperparameters, and system settings are defined here.

## Table of Contents

1. [Game Configuration](#game-configuration)
2. [Training Hyperparameters](#training-hyperparameters)
3. [DQN Agent Parameters](#dqn-agent-parameters)
4. [Model Architecture](#model-architecture)
5. [File Paths and Saving](#file-paths-and-saving)
6. [Strategic Play](#strategic-play)
7. [Inheritance & Class Structure](#inheritance--class-structure)

---

## Game Configuration

### Card and Player Settings

```python
ARRAY_LENGTH = 52  # Full 52-card deck (13 ranks × 4 suits)
NUM_PLAYERS = 4    # Number of players in the game
```

**ARRAY_LENGTH** is the fundamental constant that defines:
- The size of state arrays for tracking cards
- The number of cards in the deck (52)
- The dimensionality of card-related inputs

### Player Positions

```python
NORTH = 0  # Position 0
EAST = 1   # Position 1  
SOUTH = 2  # Position 2
WEST = 3   # Position 3
```

**Team Structure:**
- **Team 1 (North-South):** Positions 0 and 2 - DQN agents
- **Team 2 (East-West):** Positions 1 and 3 - Strategic play

### Agent Configuration

```python
DQN_AGENT_POSITIONS = [NORTH, SOUTH]      # Positions 0 and 2
EW_STRATEGY_POSITIONS = [EAST, WEST]      # Positions 1 and 3
```

---

## Training Hyperparameters

### Default Training Configuration

```python
DEFAULT_NUM_GAMES = 1000           # Number of training episodes
DEFAULT_EPSILON = 1.0              # Initial exploration rate (100%)
DEFAULT_EPSILON_DECAY = 0.996      # Exploration decay per episode
DEFAULT_MIN_EPSILON = 0.001        # Minimum exploration rate (0.1%)
```

**Epsilon-Greedy Strategy:**
- Starts at 100% random exploration
- Decays by 0.4% per episode
- Eventually stabilizes at 0.1% exploration

### Gamma Values (Discount Factor)

```python
DEFAULT_GAMMA_VALUES = [0.99, 0.95, 0.90, 0.85]
```

**Purpose:** Different gamma values for experimenting with future reward importance:
- **0.99** - Very patient, values long-term rewards highly
- **0.95** - Balanced between immediate and future rewards
- **0.90** - Prefers near-term rewards
- **0.85** - Focuses more on immediate rewards

### Model Saving

```python
DEFAULT_SAVE_EVERY = 500  # Save models every N episodes
```

### Legacy Parameters

```python
LEGACY_EPISODES = 250           # Old training episodes count
LEGACY_EPSILON_DECAY = 0.99     # Old epsilon decay rate
```

These are kept for backward compatibility with older training scripts.

---

## DQN Agent Parameters

### Replay Memory

```python
REPLAY_MEMORY_SIZE = 100        # Max transitions to keep
MIN_REPLAY_MEMORY_SIZE = 100    # Min transitions before training starts
MINIBATCH_SIZE = 32             # Batch size for training
```

**Experience Replay:**
- Stores last 100,000 transitions
- Requires at least 100 transitions before training
- Trains on random batches of 32 for stability

### Target Network

```python
UPDATE_TARGET_EVERY = 5  # Update target network every N terminal states
```

**Target Network:** Used for stable Q-learning, updated every 5 games.

### Model Configuration

```python
MODEL_NAME = 'smalle'        # Model identifier
MIN_REWARD = -200            # Minimum reward threshold
MEMORY_FRACTION = 0.35       # GPU memory fraction (if available)
```

### Default Gamma

```python
DEFAULT_GAMMA = 0.99  # Default discount factor if not using gamma array
```

---

## Model Architecture

### State Dimensions

```python
STATE_SIZE = 372  # Total state size for 52-card game
ACTION_SIZE = 13  # Cards per player
```

**State Size Calculation:** (52 × 7) + 4 + 4 = 372
- 52 × 2 = 104 (cards_array + round_array)
- 52 + 4 = 56 (hand_array + player_array)
- 52 × 4 = 208 (tracking for 4 players)
- 4 (score_array)

### Input Dimensions for Multi-Input Network

```python
GAME_INPUT_SIZE = ARRAY_LENGTH * 2      # 104: cards + round
PLAYER_INPUT_SIZE = ARRAY_LENGTH + 4    # 56: hand + player
TRACKING_INPUT_SIZE = ARRAY_LENGTH * 4  # 208: all players tracking
SCORE_INPUT_SIZE = 4                    # 4: scores
```

### Network Architecture

```python
HIDDEN_LAYER_1_SIZE = 128    # First hidden layer
HIDDEN_LAYER_2_SIZE = 64     # Second hidden layer
HIDDEN_LAYER_3_SIZE = 32     # Third hidden layer
DROPOUT_RATE = 0.35          # Dropout for regularization
```

**Network Structure:**
```
[Game Input (104)] ────┐
[Player Input (56)] ────┤
[Tracking Input (208)]──┼──→ Concatenate ──→ [Dense 128] ──→ [Dropout 0.35]
[Score Input (4)] ──────┘                     ↓
                                          [Dense 64] ──→ [Dropout 0.35]
                                              ↓
                                          [Dense 32] ──→ [Dropout 0.35]
                                              ↓
                                          [Output 13] (Q-values)
```

---

## File Paths and Saving

### Directories

```python
WEIGHTS_DIR = "Weights"  # Directory for .h5 weight files
MODELS_DIR = "Models"    # Directory for .keras model files
```

### File Name Patterns

```python
AGENT_WEIGHTS_PATTERN = "{dir}/agent_player_{player}_ep{episode}.weights.h5"
FULL_AGENT_PATTERN = "{dir}/full_agent_player_{player}_ep{episode}.keras"
```

**Usage Example:**
```python
# Save weights for player 0 at episode 500
filename = AGENT_WEIGHTS_PATTERN.format(
    dir=WEIGHTS_DIR, 
    player=0, 
    episode=500
)
# Result: "Weights/agent_player_0_ep500.weights.h5"
```

---

## Strategic Play

### EW Strategy Configuration

```python
EW_RANDOM_PLAY_PROBABILITY = 0.2  # 20% random, 80% strategic
```

**East-West Strategy:** Uses bridge-like strategic rules 80% of the time, plays randomly 20% of the time for unpredictability.

---

## Inheritance & Class Structure

### Module Reference

```python
BASE_CLASSES_MODULE = "base_classes"
```

### Agent Types

```python
AGENT_TYPE_DQN = "DQN"           # Deep Q-Network learning agent
AGENT_TYPE_STRATEGY = "STRATEGY"  # Rule-based strategy agent
AGENT_TYPE_RANDOM = "RANDOM"      # Random play agent
```

### Strategy Types

```python
STRATEGY_EW = "EW_STRATEGY"          # East-West bridge-like strategy
STRATEGY_RANDOM = "RANDOM_STRATEGY"  # Random play strategy
```

### Player Types

```python
PLAYER_TYPE_HUMAN = "HUMAN"        # Human player
PLAYER_TYPE_AI = "AI"              # AI player (DQN)
PLAYER_TYPE_STRATEGY = "STRATEGY"  # Strategy-based player
```

**Usage:** These constants can be used for:
- Runtime type checking
- Configuration files
- Player creation factories
- Logging and analytics

---

## Customization Guide

### Adjusting Training Speed

**Faster Training:**
```python
DEFAULT_NUM_GAMES = 500          # Fewer episodes
DEFAULT_EPSILON_DECAY = 0.99     # Faster exploration decay
DEFAULT_SAVE_EVERY = 100         # Save more frequently
```

**Better Performance:**
```python
DEFAULT_NUM_GAMES = 5000         # More episodes
DEFAULT_EPSILON_DECAY = 0.999    # Slower exploration decay
MINIBATCH_SIZE = 64              # Larger batches
```

### Adjusting Network Capacity

**Larger Network:**
```python
HIDDEN_LAYER_1_SIZE = 256
HIDDEN_LAYER_2_SIZE = 128
HIDDEN_LAYER_3_SIZE = 64
DROPOUT_RATE = 0.3
```

**Smaller Network:**
```python
HIDDEN_LAYER_1_SIZE = 64
HIDDEN_LAYER_2_SIZE = 32
HIDDEN_LAYER_3_SIZE = 16
DROPOUT_RATE = 0.4
```

### Adjusting Learning Behavior

**More Exploration:**
```python
DEFAULT_EPSILON = 1.0
DEFAULT_EPSILON_DECAY = 0.999
DEFAULT_MIN_EPSILON = 0.05
```

**More Exploitation:**
```python
DEFAULT_EPSILON = 0.5
DEFAULT_EPSILON_DECAY = 0.99
DEFAULT_MIN_EPSILON = 0.001
```

---

## Best Practices

1. **Never modify constants in other files** - Always import from `constants.py`
2. **Use constants instead of magic numbers** - Improves readability
3. **Document changes** - Explain why a constant was changed
4. **Test after changes** - Ensure system still works
5. **Keep related constants together** - Use the section structure

## Example Usage in Code

```python
from constants import (
    ARRAY_LENGTH,
    NUM_PLAYERS,
    DQN_AGENT_POSITIONS,
    DEFAULT_GAMMA_VALUES,
    HIDDEN_LAYER_1_SIZE,
    WEIGHTS_DIR
)

# Create input size
input_size = (ARRAY_LENGTH * 7) + 4 + 4

# Create agents with proper gamma values
agents = [
    DQNAgent(input_size, gamma=DEFAULT_GAMMA_VALUES[i])
    if i in DQN_AGENT_POSITIONS else None
    for i in range(NUM_PLAYERS)
]

# Save model
filename = f"{WEIGHTS_DIR}/agent_player_0_ep1000.weights.h5"
agent.save_agent(filename)
```

---

## Summary

The `constants.py` file provides:
- ✅ **Centralized configuration** - One place for all settings
- ✅ **Type safety** - Constants prevent accidental modifications
- ✅ **Documentation** - Clear comments explain each constant
- ✅ **Maintainability** - Easy to adjust and experiment
- ✅ **Consistency** - Same values used across entire codebase

For more details on the inheritance structure, see [INHERITANCE_STRUCTURE.md](INHERITANCE_STRUCTURE.md).
