# Reward System Documentation

## Overview

The reward system is designed to train only the DQN agents at positions 0 (North) and 2 (South), which form Team 1. Positions 1 (East) and 3 (West) use strategic rule-based play and do not receive rewards.

## 📊 Visual Guides

For a visual understanding of how rewards work in different scenarios, see:
- **[Reward Visualizations](reward_visualizations/)** - Comprehensive graphs showing:
  - Reward breakdowns for winning vs losing scenarios
  - Heatmaps of all agent/partner contribution combinations
  - Per-trick reward timelines
  - Impact of team bonus on different game outcomes

## Reward Structure

### Trick-Level Rewards

When a trick is completed (4 cards played):

| Outcome | Reward | Description |
|---------|--------|-------------|
| Agent wins trick | **+1.0** | The agent played the winning card |
| Partner wins trick | **+0.9** | The agent's partner (other agent) won the trick |
| Opponent wins trick | **-1.1** | A non-agent player (East or West) won the trick |

This reward structure encourages:
- Winning tricks directly (+1.0)
- Supporting the partner to win (+0.9)
- Avoiding letting opponents win (-1.1)

### Trump Play Penalties (New)

Agents receive penalties for suboptimal trump usage to encourage strategic play:

| Violation | Penalty Constant | Description |
|-----------|------------------|-------------|
| **Not using trump when should** | `TRUMP_NOT_USED_PENALTY` | Agent has trump cards but doesn't use them when opponent is winning and partner is not winning |
| **Trump overplay** | `TRUMP_OVERPLAY_PENALTY` | Agent uses unnecessarily high trump when a lower trump would win |
| **Taking trick from partner** | `PARTNER_OVERPLAY_PENALTY` | Agent plays a higher card to take a trick that partner was already winning |

**Note**: Stronger (more negative) penalties accelerate learning of proper trump usage but may increase initial training variance.

#### Penalty Details

**1. Not Using Trump**
- Applied when:
  - Opponent is currently winning the trick
  - Agent has trump cards in hand
  - Partner is not winning
  - Agent plays a non-trump card when unable to follow suit
- Example: Opponent leads Ace of Hearts, agent has no hearts but has Spades (trump), plays Diamond instead
- **Impact**: Higher penalty values punish this mistake more severely, encouraging agents to use trump appropriately

**2. Trump Overplay**
- Applied when:
  - Agent plays a trump card
  - Agent had lower trump cards that would also win
  - Works even when overtrumping partner unnecessarily
- Example: Opponent plays 5 of Hearts, agent has 2, 5, and King of Spades, plays King instead of 2
- **Impact**: Penalty teaches card conservation; adjust magnitude based on how critical this is for your strategy

**3. Taking Trick from Partner**
- Applied when:
  - Partner is currently winning the trick
  - Agent plays a higher card that beats partner's card
  - Agent had lower cards available that wouldn't win
- Example: Partner plays King of Hearts (winning), agent plays Ace of Hearts instead of lower card
- **Note**: Typically the strongest penalty since it's most wasteful - team was already going to win
- **Impact**: Adjust relative to other penalties based on importance of partner coordination

These penalties teach agents:
- When to use trump cards strategically
- To conserve high cards when not needed
- To avoid competing with their partner
- To coordinate effectively as a team

### Per-Card Rewards (Configurable)

Agents can receive a small reward on each card play when their choice matches what the EW strategy would play:

| Outcome | Reward Constant | Description |
|---------|-----------------|-------------|
| Match EW strategy | `PER_CARD_EW_STRATEGY_REWARD` | Agent plays the same card EW strategy would choose |
| Differ from EW strategy | No penalty | No penalty for playing differently |

This feature:
- Can be disabled by setting `enable_per_card_reward=False` in the trainer
- Can be configured via `ENABLE_PER_CARD_REWARD` in `constants.py`
- Reward value controlled by `PER_CARD_EW_STRATEGY_REWARD` constant
- **Impact**: Higher values provide stronger immediate feedback but may cause agents to mimic EW strategy too closely

**Note**: This feature is designed to be removable so the model doesn't become a copy of the EW strategy.

### Game-Level Rewards (Updated)

When the game ends (all cards played), reward is calculated using:

**Formula**: `((13 + (tricks_won - 13)) / 10) ** (1 + (tricks_won / 13)) + team_bonus`

Where:
- `max_tricks = CARDS_PER_PLAYER` (number of cards each player has, typically 13)
- `team_bonus` = +2 if team total tricks meets winning threshold (≥7 tricks), otherwise 0

**Example reward progression** (assuming 13 cards per player and team bonus of +2 for winning):

| Agent Tricks Won | Base Reward | With Team Bonus (≥7 team tricks) |
|------------------|-------------|----------------------------------|
| 0 | 0.50 | 2.50 (if team wins) |
| 3 | 0.65 | 2.65 (if team wins) |
| 5 | 0.76 | 2.76 (if team wins) |
| 7 | 0.91 | 2.91 (if team wins) |
| 10 | 1.21 | 3.21 (if team wins) |
| 13 | 2.00 | 4.00 (if team wins) |

This reward structure:
- Provides granular feedback about individual agent performance
- Applies a diminishing multiplier `(1 - tricks/max_tricks)` to scale rewards
- Awards team bonus to both agents if their team wins (controlled by threshold)
- Uses `END_GAME_REWARD_MULTIPLIER` constant for additional scaling
- **Impact**: Adjust `END_GAME_REWARD_MULTIPLIER` to change relative importance of end-game rewards vs per-trick rewards

### Model Save Threshold

Models are only saved during training if the average reward over recent episodes exceeds the threshold set by `MODEL_SAVE_REWARD_THRESHOLD`. This prevents saving poorly performing models.

**Timing**: The check is performed every `MODEL_SAVE_CHECK_EVERY` games, but only after `MODEL_SAVE_MIN_GAMES` have been played.
**Impact**: 
- Lower (more negative) threshold means more models saved, including mediocre ones
- Higher (less negative) threshold means only better models saved, but may miss intermediate progress

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

Reward constants in `constants.py` (values determined by code, not documentation):
- `ENABLE_PER_CARD_REWARD`: Boolean flag to enable/disable per-card rewards
- `PER_CARD_EW_STRATEGY_REWARD`: Reward magnitude for matching EW strategy
  - **Effect**: Higher values increase imitation of EW strategy; lower values allow more independent learning
- `END_GAME_REWARD_MULTIPLIER`: Scaling factor for end-game rewards
  - **Effect**: Increase to emphasize long-term performance; decrease to focus on per-trick rewards
- `MODEL_SAVE_REWARD_THRESHOLD`: Minimum average reward required to save model
  - **Effect**: More negative values save more models; less negative values save only better models
- `TRUMP_NOT_USED_PENALTY`: Penalty for not using trump when optimal
  - **Effect**: More negative values punish this mistake more severely
- `TRUMP_OVERPLAY_PENALTY`: Penalty for using unnecessarily high trump
  - **Effect**: More negative values emphasize card conservation
- `PARTNER_OVERPLAY_PENALTY`: Penalty for taking trick from winning partner
  - **Effect**: More negative values emphasize partner coordination
- `MODEL_SAVE_CHECK_EVERY`: Frequency of model save checks
- `MODEL_SAVE_MIN_GAMES`: Minimum games before starting model save checks

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
