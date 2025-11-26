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

### Trump Play Penalties (New)

Agents receive penalties for suboptimal trump usage to encourage strategic play:

| Violation | Penalty | Description |
|-----------|---------|-------------|
| **Not using trump when should** | **-7.0** | Agent has trump cards but doesn't use them when opponent is winning and partner is not winning |
| **Trump overplay** | **-5.0** | Agent uses unnecessarily high trump when a lower trump would win |
| **Taking trick from partner** | **-8.0** | Agent plays a higher card to take a trick that partner was already winning |

#### Penalty Details

**1. Not Using Trump (-7.0)**
- Applied when:
  - Opponent is currently winning the trick
  - Agent has trump cards in hand
  - Partner is not winning
  - Agent plays a non-trump card when unable to follow suit
- Example: Opponent leads Ace of Hearts, agent has no hearts but has Spades (trump), plays Diamond instead

**2. Trump Overplay (-5.0)**
- Applied when:
  - Agent plays a trump card
  - Agent had lower trump cards that would also win
  - Works even when overtrumping partner unnecessarily
- Example: Opponent plays 5 of Hearts, agent has 2, 5, and King of Spades, plays King instead of 2

**3. Taking Trick from Partner (-8.0)**
- Applied when:
  - Partner is currently winning the trick
  - Agent plays a higher card that beats partner's card
  - Agent had lower cards available that wouldn't win
- Example: Partner plays King of Hearts (winning), agent plays Ace of Hearts instead of lower card
- **Most severe penalty** because it's wasteful - team was already going to win the trick

These penalties teach agents:
- When to use trump cards strategically
- To conserve high cards when not needed
- To avoid competing with their partner
- To coordinate effectively as a team

### Per-Card Rewards (Configurable)

Agents can receive a small reward on each card play when their choice matches what the EW strategy would play:

| Outcome | Reward | Description |
|---------|--------|-------------|
| Match EW strategy | **+0.2** | Agent plays the same card EW strategy would choose [+0.2 EW match bonus] |
| Differ from EW strategy | **0** | No penalty for playing differently |

This feature:
- Can be disabled by setting `enable_per_card_reward=False` in the trainer
- Can be configured via `ENABLE_PER_CARD_REWARD` in `constants.py`
- Uses `PER_CARD_EW_STRATEGY_REWARD = 0.2` for the reward amount

**Note**: This feature is designed to be removable so the model doesn't become a copy of the EW strategy.

### Game-Level Rewards (Updated)

When the game ends (all cards played), reward is calculated using:

**Formula**: `(tricks_won - max_tricks) × (1 - tricks_won / max_tricks) + team_bonus`

Where:
- `max_tricks = 13` (CARDS_PER_PLAYER)
- `team_bonus = +2` if team total tricks >= 7 (wins the game), otherwise 0

| Agent Tricks Won | Multiplier | Base Reward | With Team Bonus (if applicable) |
|------------------|------------|-------------|--------------------------------|
| 13 (max) | 0.00 | **0** | +2 (team wins) |
| 10 | 0.23 | **-0.69** | +1.31 (team wins) |
| 7 | 0.46 | **-2.77** | depends on team total |
| 3 | 0.77 | **-7.69** | depends on team total |
| 0 | 1.00 | **-13** | -13 (team likely loses) |

This reward structure:
- Provides granular feedback about individual agent performance
- Applies a diminishing multiplier `(1 - tricks/13)` to scale rewards
- Awards +2 bonus to both agents if their team wins 7 or more tricks total (wins the game)
- Uses `END_GAME_REWARD_MULTIPLIER` constant for additional scaling (default: 1.0)

### Model Save Threshold

Models are only saved during training if the average reward over the last 100 episodes exceeds **-5.5**. This prevents saving poorly performing models.

**Timing**: The check is performed every 25 games, but only after 200 games have been played.

- `MODEL_SAVE_REWARD_THRESHOLD = -5.5`: Minimum average reward to save model
- `MODEL_SAVE_CHECK_EVERY = 25`: Check every 25 games
- `MODEL_SAVE_MIN_GAMES = 200`: Only start checking after 200 games
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
- Trump penalties: `calculate_trump_penalty()` method
- Per-card rewards: `calculate_per_card_reward()` method
- Trick-level rewards: `_get_game_state()` method
- Game-level rewards: `step()` method

### Constants

Reward constants in `constants.py`:
- `ENABLE_PER_CARD_REWARD`: Enable/disable per-card rewards (default: True)
- `PER_CARD_EW_STRATEGY_REWARD`: Reward for matching EW strategy (default: 0.2)
- `END_GAME_REWARD_MULTIPLIER`: Multiplier for end-game reward (default: 1.0)
- `MODEL_SAVE_REWARD_THRESHOLD`: Minimum average reward to save model (default: -0.5)
- **`TRUMP_NOT_USED_PENALTY`**: Penalty for not using trump when should (default: -7.0)
- **`TRUMP_OVERPLAY_PENALTY`**: Penalty for using unnecessarily high trump (default: -5.0)
- **`PARTNER_OVERPLAY_PENALTY`**: Penalty for taking trick from winning partner (default: -8.0)

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
