# Classic DQN Models

This directory stores full model files (`.keras`) for the classic DQN agent training approach.

## File Naming Convention

Files are saved with the following pattern:
```
full_agent_player_{player}_ep{episode}_avgR{avg_reward}_{timestamp}.keras
```

Example:
```
full_agent_player_0_ep1000_avgR-3.45_20251201_132505.keras
```

## Training

To train a classic DQN agent, run:
```bash
python -m Whist.training.model_training
```
