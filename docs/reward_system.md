# Reward System Documentation

## Overview

The reward system is designed to train only the DQN agents at positions 0 (North) and 2 (South), which form Team 1. Positions 1 (East) and 3 (West) use strategic rule-based play and do not receive rewards.

## Reward Structure

### Trick-Level Rewards

When a trick is completed (4 cards played):

| Outcome | Reward | Description |
|---------|--------|-------------|
| Agent wins trick | **+1** | The agent played the winning card |
| Partner wins trick | **+0.8** | The agent's partner (other agent) won the trick |
| Opponent wins trick | **-1** | A non-agent player (East or West) won the trick |

This reward structure encourages:
- Winning tricks directly (+1)
- Supporting the partner to win (+0.8)
- Avoiding letting opponents win (-1)

### Per-Card Rewards (New, Configurable)

Agents can receive small rewards/penalties on each card play based on whether their choice matches what the EW strategy would play:

| Outcome | Reward | Description |
|---------|--------|-------------|
| Match EW strategy | **+0.1** | Agent plays the same card EW strategy would choose |
| Differ from EW strategy | **-0.1** | Agent plays a different card |

This feature:
- Can be disabled by setting `enable_per_card_reward=False` in the trainer
- Can be configured via `ENABLE_PER_CARD_REWARD` in `constants.py`
- Uses `PER_CARD_EW_STRATEGY_REWARD` and `PER_CARD_EW_STRATEGY_PENALTY` for customization

**Note**: This feature is designed to be removable so the model doesn't become a copy of the EW strategy.

### Game-Level Rewards (Updated)

When the game ends (all cards played), reward is now based on `(tricks_won - max_tricks)`:

| Agent Tricks Won | End-Game Reward | Description |
|------------------|-----------------|-------------|
| 13 (max) | **0** | Perfect performance |
| 10 | **-3** | Good performance |
| 7 | **-6** | Average performance |
| 3 | **-10** | Below average |
| 0 | **-13** | Worst case |

This reward structure:
- Provides granular feedback about individual agent performance
- Ranges from -13 (won 0 tricks) to 0 (won all 13 tricks)
- Uses `END_GAME_REWARD_MULTIPLIER` constant for scaling (default: 1.0)

### Model Save Threshold

Models are only saved during training if the average reward over the last 100 episodes exceeds **-5.5**. This prevents saving poorly performing models.

- Configurable via `MODEL_SAVE_REWARD_THRESHOLD` in `constants.py`
- Can be overridden in trainer constructor with `model_save_threshold` parameter

## Agent Positions

- **Position 0 (North)**: DQN Agent - receives rewards
- **Position 1 (East)**: EW Strategy - does NOT receive rewards
- **Position 2 (South)**: DQN Agent - receives rewards  
- **Position 3 (West)**: EW Strategy - does NOT receive rewards

Team 1 (North-South) are partners and share the goal of winning tricks together.

## Monitoring

The reward system includes comprehensive monitoring capabilities:

### Logging Levels

Set the monitoring level to control verbosity:

```python
game.set_monitoring_level('DEBUG')  # Detailed trick-by-trick rewards
game.set_monitoring_level('INFO')   # Game-ending rewards only (default)
game.set_monitoring_level('WARNING') # Errors and warnings only
```

### Reward Statistics

Track cumulative statistics during an episode:

```python
stats = game.get_reward_stats()
# Returns:
# {
#     'agent_0_total': <cumulative reward for agent 0>,
#     'agent_2_total': <cumulative reward for agent 2>,
#     'agent_0_wins': <tricks won by agent 0>,
#     'agent_2_wins': <tricks won by agent 2>,
#     'tricks_completed': <total tricks completed>,
#     'per_card_rewards': <cumulative per-card rewards given>
# }
```

Statistics are automatically reset when `game.reset()` is called at the start of each episode.

## Implementation Details

### Location

The reward system is implemented in `whist.py`:
- Per-card rewards: `calculate_per_card_reward()` method
- Trick-level rewards: `_get_game_state()` method
- Game-level rewards: `step()` method

### Constants

Reward constants in `constants.py`:
- `ENABLE_PER_CARD_REWARD`: Enable/disable per-card rewards (default: True)
- `PER_CARD_EW_STRATEGY_REWARD`: Reward for matching EW strategy (default: 0.1)
- `PER_CARD_EW_STRATEGY_PENALTY`: Penalty for not matching (default: -0.1)
- `END_GAME_REWARD_MULTIPLIER`: Multiplier for end-game reward (default: 1.0)
- `MODEL_SAVE_REWARD_THRESHOLD`: Minimum average reward to save model (default: -5.5)

### Training Integration

The rewards are used in training files (`training_logic.py` and `training_embedded.py`):
- Per-card rewards are calculated before each step for agent positions
- Trick rewards are combined with per-card rewards in replay memory
- Model saving checks the average reward against the threshold

## Testing

Run the test script to verify the reward system:

```bash
python -m testing.test_new_reward_system
```

The test verifies:
1. End-game reward is based on (tricks_won - max_tricks)
2. Per-card rewards work based on EW strategy matching
3. Per-card rewards can be disabled
4. Model save threshold is correctly set to -5.5
5. Per-card rewards are correctly tracked in stats

## Example Output

```
Game ended. Team 1 Score: 5, Team 2 Score: 8.
Agent 0 end reward: -12.0 (tricks: 1/13)
Agent 2 end reward: -9.0 (tricks: 4/13)
```
