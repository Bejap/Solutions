# Embedded DQN Models

This directory stores full model files (`.keras`) for the embedded DQN agent training approach.

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

To train an embedded DQN agent, run:
```bash
python -m Whist.training.embedded.model_training_embedded
```

## Features

- Timestamped filenames for easy tracking
- Average reward included in filename
- Automatic saving when performance threshold is met
- Works with any card count (9, 11, or 13 cards per player)
- Compatible with prioritized experience replay
