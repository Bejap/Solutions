# Deep Simple Whist - DQN Implementation

**🎉 Recently Refactored: Now featuring comprehensive OOP inheritance structure!**

This repository contains a Deep Q-Network (DQN) implementation for playing the Whist card game. The project uses reinforcement learning to train agents to play the game effectively.

## 🆕 Recent Major Refactoring

The codebase has been completely refactored to implement a comprehensive object-oriented inheritance structure:

- **6 Abstract Base Classes** providing clear contracts for all components
- **Complete Inheritance Hierarchy** with proper polymorphism
- **Enhanced constants.py** with bug fixes (ARRAY_LENGTH: 13→52, STATE_SIZE: 91→372)
- **45K+ characters of documentation** including architecture guides and examples
- **Enterprise-grade code quality** with full code review

📖 **See the transformation**: [BEFORE_AFTER_COMPARISON.md](BEFORE_AFTER_COMPARISON.md)  
📖 **Architecture guide**: [INHERITANCE_STRUCTURE.md](INHERITANCE_STRUCTURE.md)  
📖 **Constants reference**: [CONSTANTS_GUIDE.md](CONSTANTS_GUIDE.md)  
🎮 **Try it**: `python demo_inheritance.py`

## Project Overview

This is a machine learning project that implements a Deep Q-Network (DQN) agent to play Whist, a classic trick-taking card game. The implementation includes:

- **Game Engine**: Full implementation of Whist game logic
- **DQN Agent**: Deep reinforcement learning agent using TensorFlow
- **Training System**: Optimized training loop with epsilon-greedy exploration
- **Model Management**: Save/load trained models
- **Visualization**: Plot training progress and model performance
- **Testing**: Test trained agents against each other

## Game Configuration

The game uses a **full 52-card deck** with **Spades as the locked trump suit**:

- **Full standard deck**: 13 ranks × 4 suits = 52 cards
- **Suits**: Clubs, Diamonds, Hearts, and Spades
- **Trump suit**: Spades (locked, cannot be changed)
- **Cards per player**: 13 (52 ÷ 4 players)
- **Tricks per game**: 13 (one per card in each player's hand)

### Trump Rules

- **Trump beats everything**: Any Spade beats any non-Spade card
- **Highest trump wins**: When multiple Spades are played, highest rank wins
- **No trump**: When no Spades played, highest card in led suit wins
- **Following suit**: Players must follow the led suit if able

See `docs/trump_system.md` for complete trump rules and examples.

## Team Structure

The game uses a traditional 4-player setup with two teams:

- **Team 1 (North-South)**: Players at positions 0 and 2 - controlled by DQN agents
- **Team 2 (East-West)**: Players at positions 1 and 3 - use strategic rule-based play (80% of the time) with 20% randomness

North and South agents learn to cooperate as partners on the same team. East and West provide consistent, challenging opponents using bridge-like playing strategies. See `docs/team_structure.md` for more details.

## Reward System

The game uses a sophisticated reward structure to train the agents:

- **+1** for winning a trick
- **+0.8** for partner winning a trick  
- **-1** for losing a trick
- **+10** for winning the game
- **-20** for losing the game

Only the two DQN agents (positions 0 and 2) receive rewards. The system includes comprehensive monitoring and statistics tracking. See `docs/reward_system.md` for complete details.

## Files

### Core Architecture (NEW!)
- `base_classes.py` - Abstract base classes for all components (BaseAgent, BasePlayer, BaseGame, BaseStrategy, BaseCard, BaseDeck)
- `demo_inheritance.py` - Demonstration of the inheritance structure

### Game Implementation
- `whist_game.py` - Core game classes (Card, Deck, Player) - now with inheritance
- `whist.py` - Main Whist game environment with reward system - now extends BaseGame
### Game Implementation
- `whist_game.py` - Core game classes (Card, Deck, Player) - now with inheritance
- `whist.py` - Main Whist game environment with reward system - now extends BaseGame
- `simple_whist_DQN.py` - DQN agent implementation - now extends BaseAgent
- `ew_strategy.py` - Strategic rule-based player - now extends BaseStrategy
- `constants.py` - Centralized constants (enhanced with inheritance config, bugs fixed)

### Training & Testing
### Training & Testing
- `training_logic.py` - WhistTrainer class with training logic
- `model_training.py` - Entry point script to run training
- `model_testing.py` - Test trained models
- `loading_model.py` - Load and use saved models
- `model_plotting.py` - Visualization tools
- `game.py` - Simple game initialization script

### Test Files
- `test_reward_system.py` - Verify reward distribution
- `test_trump_system.py` - Verify trump system
- Test files for game mechanics

### Directories
- `Weights/` - Model weights (*.h5 files)
- `Models/` - Full models (*.keras files)
- `docs/` - Additional documentation

### Documentation (NEW!)
- `INHERITANCE_STRUCTURE.md` - Complete architecture guide (10K chars)
- `CONSTANTS_GUIDE.md` - Constants reference (9K chars)
- `REFACTORING_SUMMARY.md` - High-level overview (10K chars)
- `BEFORE_AFTER_COMPARISON.md` - Transformation details (11K chars)
- `docs/` - Additional technical documentation

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All project constants are centralized in `constants.py`, including:

- **Game Configuration**: Number of cards (13), player positions, team setup
- **Training Hyperparameters**: Learning rates, epsilon decay, gamma values
- **DQN Parameters**: Replay memory size, batch size, update frequency
- **Model Architecture**: Network layer sizes, dropout rates, input dimensions

To customize training behavior, modify values in `constants.py` or pass parameters when instantiating `WhistTrainer`:

```python
from training_logic import WhistTrainer

trainer = WhistTrainer(
    num_games=2000,        # Override default
    epsilon=0.95,          # Custom starting exploration
    save_every=1000        # Save less frequently
)
```

## Usage

### Training a Model

```python
python model_training.py
```

### Testing a Model

```python
python model_testing.py
```

### Loading a Saved Model

```python
python loading_model.py
```

## Project Structure

The DQN agent uses a multi-input neural network architecture that processes:
- Game state (cards played in current round)
- Player's hand
- Tracking information (cards played throughout the game)
- Score information

The agent learns through experience replay and uses a target network for stable training. The network architecture includes:
- 3 hidden layers (128, 64, and 32 units) with dropout (rate=0.35)
- **Input dimensions for 52-card game** (full deck with trump)
- Input size: (52 * 7) + 4 + 4 = **372 features**
- Approximately ~56,000 parameters (adjusted for 52-card deck)
- Batch size of 32 for stable gradient updates

### Performance Optimizations

This version includes several optimizations for faster training:
- **Dynamic action space** based on hand size (no hardcoded values)
- **Increased batch size** (32 instead of 8) for more stable learning
- **Reduced replay memory threshold** (100 instead of 1000) for faster startup
- **Improved trick counting** using proper trick counter instead of action counter
- **Better action selection** using max Q-value from valid actions
- **Optimized predictions** with explicit batch_size parameters

For detailed architecture information, see `docs/architecture.md`.

## License

This project is part of a solutions repository for educational purposes.
