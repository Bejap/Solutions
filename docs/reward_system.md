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

### Game-Level Rewards

When the game ends (all cards played):

| Outcome | Reward | Description |
|---------|--------|-------------|
| Team 1 wins game | **+10** | Agents' team has more total tricks than opponents |
| Team 1 loses game | **-20** | Opponents' team has more total tricks |
| Tie game | **0** | Equal number of tricks won by both teams |

The asymmetric game-ending rewards (-20 for loss vs +10 for win) encourages:
- Strong penalty for losing to motivate better play
- Moderate reward for winning to balance the learning signal

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
#     'tricks_completed': <total tricks completed>
# }
```

Statistics are automatically reset when `game.reset()` is called at the start of each episode.

## Implementation Details

### Location

The reward system is implemented in `whist.py`:
- Trick-level rewards: `_get_game_state()` method (lines ~213-255)
- Game-level rewards: `step()` method (lines ~163-213)

### Monitoring

Monitoring capabilities include:
- Logging configuration (lines 12-14)
- Reward statistics tracking (lines 41-71)
- Methods: `set_monitoring_level()`, `get_reward_stats()`, `reset_reward_stats()`

### Training Integration

The rewards are used in `model_training.py`:
- Agents receive rewards after each step
- Rewards are stored in replay memory for training
- Statistics are logged every 100 episodes

## Testing

Run the test script to verify the reward system:

```bash
python test_reward_system.py
```

The test verifies:
1. Only agents at positions 0 and 2 receive rewards
2. Positions 1 and 3 always receive 0 rewards
3. Reward values match the expected structure
4. Monitoring functionality works correctly

## Example Output

```
Trick 1 completed. Winner: Player 0.
Agent rewards: Agent 0: 1, Agent 2: 0.8

Trick 2 completed. Winner: Player 1.  
Agent rewards: Agent 0: -1, Agent 2: -1

Game ended. Team 1 (Agents) WINS! Score: 7-6.
Agents receive +10 bonus each.
```
