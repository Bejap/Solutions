# Classic DQN Training

This directory contains training scripts and logic for the classic DQN agent approach using one-hot encoding.

## Files

- `model_training.py` - Main training script (run this to train)
- `training_logic.py` - Core training logic and WhistTrainer class
- `loading_and_continue_training.py` - Load and continue training from saved models
- `loading_model.py` - Load trained models for evaluation

## Features

- **One-hot encoding** for card and state representation
- **Prioritized experience replay** for efficient learning (enabled by default)
- **Double DQN** algorithm to reduce overestimation
- **Dueling DQN** architecture option for better value estimation
- **Timestamped models** with average reward tracking

## Usage

Train a classic DQN agent:
```bash
python -m Whist.training.classic.model_training
```

Or from the training/classic directory:
```bash
python model_training.py
```

## Model Storage

Trained models are saved with timestamps and performance metrics:
- `Weights/classic/` - Weight files: `agent_player_0_ep1000_avgR-3.45_20251201_132505.weights.h5`
- `Models/classic/` - Full model files: `full_agent_player_0_ep1000_avgR-3.45_20251201_132505.keras`

## Configuration

Key parameters (in `Whist/utils/constants.py`):
- `USE_PRIORITIZED_REPLAY = True` - Enable prioritized replay
- `REPLAY_MEMORY_SIZE = 50000` - Replay memory capacity
- `MINIBATCH_SIZE = 64` - Training batch size
- `UPDATE_TARGET_EVERY = 5` - Target network update frequency
