# Embedded DQN Training

This directory contains training scripts and logic for the embedded DQN agent approach using card embeddings.

## Files

- `model_training_embedded.py` - Main training script (run this to train)
- `training_embedded.py` - Core training logic and EmbeddedWhistTrainer class
- `loading_and_continue_training_embedded.py` - Load and continue training from saved models

## Features

- **Fixed state size** (works with 9, 11, or 13 cards per player)
- **Compact representation** (64 vs 372 dimensions for classic)
- **Learned card relationships** through embeddings
- **GPU/NPU acceleration** support for faster training
- **Prioritized experience replay** for efficient learning (enabled by default)
- **Double DQN** algorithm to reduce overestimation
- **Timestamped models** with average reward tracking

## Usage

Train an embedded DQN agent:
```bash
python -m Whist.training.embedded.model_training_embedded
```

Or from the training/embedded directory:
```bash
python model_training_embedded.py
```

## Model Storage

Trained models are saved with timestamps and performance metrics:
- `Weights/embedded/` - Weight files: `agent_player_0_ep1000_avgR-3.45_20251201_132505.weights.h5`
- `Models/embedded/` - Full model files: `full_agent_player_0_ep1000_avgR-3.45_20251201_132505.keras`

## Configuration

Key parameters (in `Whist/utils/constants.py`):
- `USE_PRIORITIZED_REPLAY = True` - Enable prioritized replay
- `PER_ALPHA = 0.6` - Prioritization exponent
- `PER_BETA_START = 0.4` - Initial importance sampling exponent
- `REPLAY_MEMORY_SIZE = 50000` - Replay memory capacity
- `CARDS_PER_PLAYER = 13` - Configurable game size

## Benefits

The embedded agent offers several advantages:
- ✓ **Flexible**: Same model works for any card count
- ✓ **Compact**: Smaller state representation
- ✓ **Faster**: Learns card relationships more efficiently
- ✓ **Scalable**: Better GPU utilization with smaller models
