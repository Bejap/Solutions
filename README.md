# Deep Simple Whist - DQN Implementation

This repository contains a Deep Q-Network (DQN) implementation for playing the Whist card game. The project uses reinforcement learning to train agents to play the game effectively.

## Project Overview

This is a machine learning project that implements a Deep Q-Network (DQN) agent to play Whist, a classic trick-taking card game. The implementation includes:

- **Game Engine**: Full implementation of Whist game logic
- **DQN Agent**: Deep reinforcement learning agent using TensorFlow
- **Training System**: Training loop with epsilon-greedy exploration
- **Model Management**: Save/load trained models
- **Visualization**: Plot training progress and model performance
- **Testing**: Test trained agents against each other

## Files

- `whist_game.py` - Core game classes (Card, Deck, Player)
- `whist.py` - Main Whist game environment implementation
- `simple_whist_DQN.py` - DQN agent implementation
- `model_training.py` - Training loop for the DQN agents
- `model_testing.py` - Test trained models
- `loading_model.py` - Load and use saved models
- `model_plotting.py` - Visualization tools for training results
- `game.py` - Simple game initialization script

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

The agent learns through experience replay and uses a target network for stable training.

## License

This project is part of a solutions repository for educational purposes.
