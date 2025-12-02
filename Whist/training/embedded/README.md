# Embedded DQN Training

This directory contains training scripts and logic for the embedded DQN agent approach using card embeddings.

## Files

- `model_training_embedded.py` - Main training script (run this to train)
- `training_embedded.py` - Core training logic and EmbeddedWhistTrainer class
- `loading_and_continue_training_embedded.py` - Load and continue training from saved models

## Features

- Fixed state size (works with 9, 11, or 13 cards per player)
- Compact representation (64 vs 372 dimensions)
- Learned card relationships through embeddings
- GPU/NPU acceleration support

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

Trained models are saved to:
- `Weights/embedded/` - Weight files (.weights.h5)
- `Models/embedded/` - Full model files (.keras)
