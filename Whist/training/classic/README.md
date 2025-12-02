# Classic DQN Training

This directory contains training scripts and logic for the classic DQN agent approach using one-hot encoding.

## Files

- `model_training.py` - Main training script (run this to train)
- `training_logic.py` - Core training logic and WhistTrainer class
- `loading_and_continue_training.py` - Load and continue training from saved models
- `loading_model.py` - Load trained models for evaluation

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

Trained models are saved to:
- `Weights/classic/` - Weight files (.weights.h5)
- `Models/classic/` - Full model files (.keras)
