# Embedded DQN Model Weights

This directory stores weight files (`.weights.h5`) for the embedded DQN agent training approach.

## File Naming Convention

Files are saved with the following pattern:
```
agent_player_{player}_ep{episode}_avgR{avg_reward}_{timestamp}.weights.h5
```

Example:
```
agent_player_0_ep1000_avgR-3.45_20251201_132505.weights.h5
```

## Training

To train an embedded DQN agent, run:
```bash
python -m Whist.training.model_training_embedded
```
