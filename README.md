# Deep Simple Whist - DQN Implementation (7-Card Version)

This repository contains a Deep Q-Network (DQN) implementation for playing the Whist card game. The project uses reinforcement learning to train agents to play the game effectively.

## Project Overview

This is a machine learning project that implements a Deep Q-Network (DQN) agent to play Whist, a classic trick-taking card game. The implementation includes:

- **Game Engine**: Full implementation of 7-card Whist game logic
- **DQN Agent**: Deep reinforcement learning agent using TensorFlow
- **Training System**: Optimized training loop with epsilon-greedy exploration
- **Model Management**: Save/load trained models
- **Visualization**: Plot training progress and model performance
- **Testing**: Test trained agents against each other

## Game Configuration

The game has been optimized to use **7 cards per player** (28 total cards: ranks 2-8 in all 4 suits):

- Each game consists of **7 tricks** (one per card)
- Faster training due to smaller state space
- Simplified learning for the DQN agents
- Each player gets 7 cards dealt at the start

## Team Structure

The game uses a traditional 4-player setup with two teams:

- **Team 1 (North-South)**: Players at positions 0 and 2 - controlled by DQN agents
- **Team 2 (East-West)**: Players at positions 1 and 3 - use strategic rule-based play (80% of the time) with 20% randomness

North and South agents learn to cooperate as partners on the same team. East and West provide consistent, challenging opponents using bridge-like playing strategies. See `docs/team_structure.md` for more details.

## Files

- `whist_game.py` - Core game classes (Card, Deck, Player)
- `whist.py` - Main Whist game environment implementation
- `simple_whist_DQN.py` - DQN agent implementation
- `ew_strategy.py` - Strategic rule-based player for East-West positions
- `model_training.py` - Training loop for the DQN agents
- `model_testing.py` - Test trained models
- `loading_model.py` - Load and use saved models
- `model_plotting.py` - Visualization tools for training results
- `game.py` - Simple game initialization script
- `docs/` - Documentation folder with architecture details, optimization suggestions and team structure info

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

The agent learns through experience replay and uses a target network for stable training. The network has been optimized for the 7-card game with:
- 2 hidden layers (64 and 32 units) with dropout (rate=0.25)
- Smaller input dimensions (7 cards instead of 13)
- Only ~7,400 parameters for faster training
- Batch size of 32 for stable gradient updates

### Performance Optimizations

This version includes several optimizations for faster training:
- **7 cards per player** instead of 13 (smaller state space)
- **Dynamic action space** based on hand size (no hardcoded values)
- **Increased batch size** (32 instead of 8) for more stable learning
- **Reduced replay memory threshold** (100 instead of 1000) for faster startup
- **Smaller neural network** (~7,400 parameters vs previous larger model)
- **Optimized predictions** with explicit batch_size parameters

Training speed: ~1.8 seconds per game on CPU

For detailed architecture information, see `docs/architecture.md`.

## License

This project is part of a solutions repository for educational purposes.
