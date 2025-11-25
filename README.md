# Deep Simple Whist - DQN Implementation

**🎉 Recently Refactored: Now featuring comprehensive OOP inheritance structure and organized folder layout!**

This repository contains a Deep Q-Network (DQN) implementation for playing the Whist card game. The project uses reinforcement learning to train agents to play the game effectively.

## 📁 Project Structure

The project is organized into clear, descriptive folders:

```
Solutions/
├── Whist/                          # Main game package
│   ├── core/                       # Core game logic
│   │   ├── whist.py               # Main game environment
│   │   ├── whist_game.py          # Card, Deck, Player classes
│   │   ├── whist_embedded.py      # Embedding-based game adapter
│   │   └── game.py                # Simple game initialization
│   │
│   ├── agents/                     # AI agents
│   │   ├── simple_whist_DQN.py    # DQN agent implementation
│   │   ├── embedded_dqn_agent.py  # Embedding-based DQN agent
│   │   └── ew_strategy.py         # East-West strategy player
│   │
│   ├── training/                   # Training scripts
│   │   ├── training_logic.py      # WhistTrainer class
│   │   ├── training_embedded.py   # EmbeddedWhistTrainer class
│   │   ├── model_training.py      # Original training entry point
│   │   ├── model_training_embedded.py  # Embedded training entry point
│   │   └── loading_model.py       # Model loading utilities
│   │
│   ├── embedding/                  # Card embedding system
│   │   └── card_embedding.py      # CardEmbedding class
│   │
│   ├── utils/                      # Utilities and constants
│   │   ├── constants.py           # Centralized constants
│   │   └── base_classes.py        # Abstract base classes
│   │
│   ├── logger/                     # Logging and visualization
│   │   ├── game_logger.py         # Game logging
│   │   └── model_plotting.py      # Training visualization
│   │
│   └── demos/                      # Demo scripts
│       ├── demo_inheritance.py    # Inheritance demonstration
│       ├── demo_features.py       # Feature demonstration
│       └── model_testing.py       # Model testing script
│
├── testing/                        # Test files
│   ├── test_reward_system.py
│   ├── test_trump_system.py
│   ├── test_follow_suit.py
│   └── ...
│
├── docs/                           # Documentation
│   ├── INHERITANCE_STRUCTURE.md
│   ├── CARD_EMBEDDING_GUIDE.md
│   ├── CONSTANTS_GUIDE.md
│   └── ...
│
├── Models/                         # Saved full models (*.keras)
├── Weights/                        # Saved model weights (*.h5)
├── plots/                          # Training plots and visualizations
├── game_logs/                      # Regular training game logs
├── game_logs_embedded/             # Embedded training game logs
└── README.md
```

## 🆕 Recent Major Updates

### Folder Reorganization
- **Whist/**: Main package with clear subfolders (core, agents, training, embedding, utils, logger, demos)
- **testing/**: All test files in one place
- **docs/**: All documentation files organized

### Object-Oriented Refactoring
- **6 Abstract Base Classes** providing clear contracts for all components
- **Complete Inheritance Hierarchy** with proper polymorphism

### Card Embedding System
- **Fixed 64-dim state** independent of card count
- **Same model works** for 9, 11, or 13 cards per player
- Run: `python -m Whist.training.model_training_embedded`

📖 **Documentation in docs/**:
- [INHERITANCE_STRUCTURE.md](docs/INHERITANCE_STRUCTURE.md) - Architecture guide  
- [CARD_EMBEDDING_GUIDE.md](docs/CARD_EMBEDDING_GUIDE.md) - Embedding system  
- [CONSTANTS_GUIDE.md](docs/CONSTANTS_GUIDE.md) - Constants reference

## Project Overview

This is a machine learning project that implements a Deep Q-Network (DQN) agent to play Whist, a classic trick-taking card game. The implementation includes:

- **Game Engine**: Full implementation of Whist game logic
- **DQN Agent**: Deep reinforcement learning agent using TensorFlow
- **Card Embeddings**: Vector-based card representations for flexible game sizes
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

### Trick-Level Rewards
- **+1** for winning a trick
- **+0.8** for partner winning a trick  
- **-1** for losing a trick

### End-Game Rewards (New)
The end-game reward is now based on `(tricks_won - max_tricks)` where `max_tricks = 13` (CARDS_PER_PLAYER). This means:
- If an agent wins 5 tricks: reward = 5 - 13 = **-8**
- If an agent wins 10 tricks: reward = 10 - 13 = **-3**
- If an agent wins 13 tricks: reward = 13 - 13 = **0** (best possible)

### Per-Card Rewards (New, Configurable)
Agents can receive small rewards/penalties on each card play based on whether their choice matches what the EW strategy would play:
- **+0.1** if agent plays the same card EW strategy would choose
- **-0.1** if agent plays a different card

This feature can be disabled by setting `enable_per_card_reward=False` in the trainer or `ENABLE_PER_CARD_REWARD=False` in constants.

### Model Save Threshold
Models are only saved if the average reward over the last 100 episodes is above **-5.5**. This prevents saving poorly performing models.

Only the two DQN agents (positions 0 and 2) receive rewards. The system includes comprehensive monitoring and statistics tracking. See `docs/reward_system.md` for complete details.

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

## Usage

### Training with Original Agent (One-Hot)

```bash
python -m Whist.training.model_training
```

### Training with Embedded Agent (Recommended)

```bash
python -m Whist.training.model_training_embedded
```

### Running Tests

```bash
python -m testing.test_reward_system
python -m testing.test_trump_system
```

### Importing in Code

```python
# Import from the organized structure
from Whist.core.whist import Whist
from Whist.agents.simple_whist_DQN import DQNAgent
from Whist.agents.embedded_dqn_agent import EmbeddedDQNAgent
from Whist.utils.constants import CARDS_PER_PLAYER, STATE_SIZE
from Whist.training.training_embedded import EmbeddedWhistTrainer
```

## Configuration

All project constants are centralized in `Whist/utils/constants.py`, including:

- **Game Configuration**: Number of cards, player positions, team setup
- **Training Hyperparameters**: Learning rates, epsilon decay, gamma values
- **DQN Parameters**: Replay memory size, batch size, update frequency
- **Model Architecture**: Network layer sizes, dropout rates, input dimensions

To customize training behavior:

```python
from Whist.training.training_embedded import EmbeddedWhistTrainer

trainer = EmbeddedWhistTrainer(
    embedding_dim=8,       # Card embedding dimensions
    num_games=2000,        # Number of training episodes
    epsilon=0.95,          # Starting exploration rate
    save_every=500,        # Save model frequency
    log_every=100,         # Log game details frequency
    log_dir='game_logs_embedded'  # Game log directory
)
trainer.train()

# Generate and save plots to 'plots/' folder
trainer.plot_results(plot_dir='plots')
```

### Output Folders

- **`plots/`**: Training visualization plots
  - `reward_over_time_*.png` - Reward progression with rolling average
  - `reward_distribution_*.png` - Histogram of rewards
  - `cumulative_reward_*.png` - Cumulative reward over training

- **`game_logs/`**: Detailed game logs for regular training
- **`game_logs_embedded/`**: Detailed game logs for embedded training

Each log includes starting hands, every card played (with decision type: Agent/Random/Strategy), trick winners, and final scores.

## License

This project is part of a solutions repository for educational purposes.
