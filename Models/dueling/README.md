# Dueling DQN Models

This directory stores full model files (`.keras`) for the dueling DQN agent training approach.

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

To train a dueling DQN agent, run:
```bash
python -m Whist.training.classic.model_training
```

Set `use_dueling_dqn=True` when creating the `WhistTrainer` instance.

## Features

- Timestamped filenames for easy tracking
- Average reward included in filename
- Automatic saving when performance threshold is met
- Separate value and advantage streams for better learning
- Compatible with prioritized experience replay
- Supports n-step returns and learning rate scheduling
