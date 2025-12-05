# Reward System Visualizations

This directory contains visual representations of how the reward system works in different game scenarios.

## Overview

The Whist DQN agent uses a sophisticated reward system that combines:
- **Trick-level rewards**: Immediate feedback for each trick won/lost
- **End-game rewards**: Performance-based rewards at game completion
- **Team bonus**: Additional reward for winning the game as a team

## Visualization Files

### 1. Reward Scenarios Breakdown (`reward_scenarios_breakdown.png`)

Shows 4 key scenarios with reward component breakdown:

- **Top Left - Team Wins (7 tricks total)**: Shows how rewards vary when the team wins but the agent contributes different amounts
- **Top Right - Team Loses (6 tricks total)**: Shows rewards when team loses with different contribution splits
- **Bottom Left - Partner Carries**: When partner wins most tricks but team still wins
- **Bottom Right - Agent Carries**: When agent wins most tricks and leads the team to victory

**Key Insights**:
- Winning more tricks individually is better, BUT team winning adds significant bonus
- Partner winning tricks still benefits the agent (+0.8 per trick)
- Team bonus of +2 can turn a mediocre individual performance into positive reward

### 2. Reward Heatmap (`reward_heatmap.png`)

A comprehensive heatmap showing total rewards for every possible combination of agent and partner trick wins.

**Features**:
- X-axis: Agent tricks won (0-13)
- Y-axis: Partner tricks won (0-13)
- Color: Green = positive reward, Red = negative reward
- Blue dashed line: Team win threshold (7 tricks)

**Key Insights**:
- Clear division at the win threshold line
- Best rewards: upper-right corner (both contribute highly AND team wins)
- Worst rewards: lower-left corner (both contribute poorly AND team loses)
- Diagonal patterns show balanced vs unbalanced contributions

### 3. Win vs Loss Comparison (`win_vs_loss_comparison.png`)

Direct side-by-side comparison of identical trick distributions but different game outcomes.

- **Left panel**: Team wins with 7 tricks total
- **Right panel**: Team loses with 6 tricks total

**Key Insights**:
- The +2 team bonus makes a dramatic difference
- Even with 0 personal tricks, winning team gives positive end-game reward
- Losing team faces significant negative end-game penalty
- Shows the importance of team coordination over individual performance

### 4. Per-Trick Timeline (`per_trick_timeline.png`)

An example game played trick-by-trick showing:
- Individual trick rewards (bars)
- Cumulative reward over time (line)
- Color coding: Green (agent wins), Light green (partner wins), Red (opponent wins)

**Example Scenario**: Agent wins 4 tricks, Partner wins 3 tricks, Opponents win 6 tricks

**Key Insights**:
- Real-time feedback helps agent learn which actions led to wins/losses
- Cumulative reward shows overall game trajectory
- Partner wins provide positive reinforcement for team play
- Opponent wins provide negative feedback to avoid similar situations

## Reward Formula

### Trick-Level Rewards
| Outcome | Reward | Meaning |
|---------|--------|---------|
| Agent wins trick | **+1.0** | Direct win |
| Partner wins trick | **+0.8** | Team win (slightly less to encourage agent to win when possible) |
| Opponent wins trick | **-1.0** | Loss (need to improve strategy) |

### End-Game Reward Formula

```python
base_reward = (agent_tricks - max_tricks) × (1 - agent_tricks / max_tricks)
team_bonus = +2.0 if team_total >= 7 else 0.0
end_game_reward = base_reward + team_bonus
```

**Components**:
- `max_tricks = 13` (cards per player)
- `agent_tricks`: Number of tricks won by the agent
- `team_total`: Agent tricks + Partner tricks
- Team wins if `team_total >= 7`

### Total Game Reward

```
Total Reward = Sum of all trick-level rewards + End-game reward
```

## Examples

### Example 1: Agent Carries Team to Victory
- Agent: 5 tricks → Trick rewards = +5.0
- Partner: 2 tricks → Trick rewards = +1.6
- Opponents: 6 tricks → Trick rewards = -6.0
- Trick total: 5 + 1.6 - 6 = **+0.6**
- End-game: (5-13) × (1-5/13) + 2 = -8 × 0.615 + 2 = **-2.92**
- **Total: 0.6 + (-2.92) = -2.32**

Despite winning, the individual performance wasn't strong enough for a highly positive reward, but team bonus prevents it from being very negative.

### Example 2: Partner Carries, Agent Contributes Minimally
- Agent: 1 trick → Trick rewards = +1.0
- Partner: 6 tricks → Trick rewards = +4.8
- Opponents: 6 tricks → Trick rewards = -6.0
- Trick total: 1 + 4.8 - 6 = **-0.2**
- End-game: (1-13) × (1-1/13) + 2 = -12 × 0.923 + 2 = **-9.08**
- **Total: -0.2 + (-9.08) = -9.28**

Partner carried the team to victory, but agent's poor individual performance results in negative reward. This teaches the agent to contribute more.

### Example 3: Both Contribute Equally to Win
- Agent: 4 tricks → Trick rewards = +4.0
- Partner: 3 tricks → Trick rewards = +2.4
- Opponents: 6 tricks → Trick rewards = -6.0
- Trick total: 4 + 2.4 - 6 = **+0.4**
- End-game: (4-13) × (1-4/13) + 2 = -9 × 0.692 + 2 = **-4.23**
- **Total: 0.4 + (-4.23) = -3.83**

Balanced contribution with team win. Still negative overall due to opponents winning more tricks.

### Example 4: Dominant Team Victory
- Agent: 6 tricks → Trick rewards = +6.0
- Partner: 6 tricks → Trick rewards = +4.8
- Opponents: 1 trick → Trick rewards = -1.0
- Trick total: 6 + 4.8 - 1 = **+9.8**
- End-game: (6-13) × (1-6/13) + 2 = -7 × 0.538 + 2 = **-1.77**
- **Total: 9.8 + (-1.77) = +8.03**

Strong individual AND team performance results in highly positive reward!

## Using These Visualizations

These visualizations help understand:
1. **Why certain strategies are rewarded**: See which scenarios give positive rewards
2. **Team vs individual balance**: Understand the trade-off between personal tricks and team success
3. **Learning trajectory**: Per-trick timeline shows how rewards accumulate
4. **Model behavior**: Helps debug why trained models make certain decisions

## Generating New Visualizations

To regenerate these visualizations with updated parameters:

```bash
python generate_reward_visualizations.py
```

The script reads constants from the reward system and generates fresh visualizations automatically.

## Related Documentation

- [reward_system.md](../reward_system.md) - Complete reward system documentation
- [reward_suggestions.md](../reward_suggestions.md) - Alternative reward system ideas
- [training_logic.py](../../Whist/training/classic/training_logic.py) - Where rewards are applied during training
