"""
Reward System Visualization

This script creates visualizations showing how rewards are calculated
in different game scenarios for the Whist DQN training.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('docs/reward_visualizations', exist_ok=True)

# Constants from the reward system
CARDS_PER_PLAYER = 13
MAX_TRICKS = CARDS_PER_PLAYER
TEAM_WIN_THRESHOLD = 7  # Team needs 7+ tricks to win

def calculate_end_game_reward(agent_tricks, partner_tricks, team_won):
    """
    Calculate end-game reward using the current formula:
    (tricks_won - max_tricks) × (1 - tricks_won / max_tricks) + team_bonus
    """
    base_reward = (agent_tricks - MAX_TRICKS) * (1 - agent_tricks / MAX_TRICKS)
    team_bonus = 2.0 if team_won else 0.0
    return base_reward + team_bonus

def calculate_total_game_reward(agent_tricks, partner_tricks):
    """Calculate total reward including trick-level rewards."""
    # Trick-level rewards (assuming rest are split between partner and opponents)
    opponent_tricks = MAX_TRICKS - (agent_tricks + partner_tricks)
    
    trick_level_reward = agent_tricks * 1.0  # +1 for each trick agent wins
    trick_level_reward += partner_tricks * 0.8  # +0.8 for each trick partner wins
    trick_level_reward += opponent_tricks * (-1.0)  # -1 for each trick opponents win
    
    # End-game reward
    team_total = agent_tricks + partner_tricks
    team_won = team_total >= TEAM_WIN_THRESHOLD
    end_game_reward = calculate_end_game_reward(agent_tricks, partner_tricks, team_won)
    
    return trick_level_reward, end_game_reward, trick_level_reward + end_game_reward

# ============================================================================
# Figure 1: Reward Components by Agent Tricks Won
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Reward System Breakdown: Different Scenarios', fontsize=16, fontweight='bold')

# Scenario 1: Team Wins (7-6 split)
ax = axes[0, 0]
agent_tricks_range = range(0, 8)
trick_rewards = []
end_game_rewards = []
total_rewards = []

for agent_tricks in agent_tricks_range:
    partner_tricks = 7 - agent_tricks  # Team wins with 7 total
    if partner_tricks < 0:
        partner_tricks = 0
    trick, end_game, total = calculate_total_game_reward(agent_tricks, partner_tricks)
    trick_rewards.append(trick)
    end_game_rewards.append(end_game)
    total_rewards.append(total)

ax.bar(agent_tricks_range, trick_rewards, label='Trick-level Rewards', alpha=0.7, color='skyblue')
ax.bar(agent_tricks_range, end_game_rewards, bottom=trick_rewards, label='End-game Bonus', alpha=0.7, color='lightgreen')
ax.plot(agent_tricks_range, total_rewards, 'ro-', linewidth=2, markersize=8, label='Total Reward')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Agent Tricks Won', fontsize=11)
ax.set_ylabel('Reward', fontsize=11)
ax.set_title('Team Wins (7 tricks total, varying split)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Scenario 2: Team Loses (6-7 split)
ax = axes[0, 1]
agent_tricks_range = range(0, 7)
trick_rewards = []
end_game_rewards = []
total_rewards = []

for agent_tricks in agent_tricks_range:
    partner_tricks = 6 - agent_tricks  # Team loses with 6 total
    if partner_tricks < 0:
        partner_tricks = 0
    trick, end_game, total = calculate_total_game_reward(agent_tricks, partner_tricks)
    trick_rewards.append(trick)
    end_game_rewards.append(end_game)
    total_rewards.append(total)

ax.bar(agent_tricks_range, trick_rewards, label='Trick-level Rewards', alpha=0.7, color='salmon')
ax.bar(agent_tricks_range, end_game_rewards, bottom=trick_rewards, label='End-game Penalty', alpha=0.7, color='lightcoral')
ax.plot(agent_tricks_range, total_rewards, 'ro-', linewidth=2, markersize=8, label='Total Reward')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Agent Tricks Won', fontsize=11)
ax.set_ylabel('Reward', fontsize=11)
ax.set_title('Team Loses (6 tricks total, varying split)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Scenario 3: Partner Carries (low agent, high partner)
ax = axes[1, 0]
scenarios = [
    (0, 7, 'Partner\nwins all'),
    (1, 6, '1-6 split'),
    (2, 5, '2-5 split'),
    (3, 4, '3-4 split'),
]
x_pos = np.arange(len(scenarios))
trick_rewards = []
end_game_rewards = []
total_rewards = []
labels = []

for agent_tricks, partner_tricks, label in scenarios:
    trick, end_game, total = calculate_total_game_reward(agent_tricks, partner_tricks)
    trick_rewards.append(trick)
    end_game_rewards.append(end_game)
    total_rewards.append(total)
    labels.append(label)

ax.bar(x_pos, trick_rewards, label='Trick-level Rewards', alpha=0.7, color='skyblue')
ax.bar(x_pos, end_game_rewards, bottom=trick_rewards, label='End-game Bonus', alpha=0.7, color='lightgreen')
ax.plot(x_pos, total_rewards, 'ro-', linewidth=2, markersize=8, label='Total Reward')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Reward', fontsize=11)
ax.set_title('Partner Carries (Team Wins, Agent Contributes Less)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Scenario 4: Agent Carries (high agent, low partner)
ax = axes[1, 1]
scenarios = [
    (7, 0, 'Agent\nwins all'),
    (6, 1, '6-1 split'),
    (5, 2, '5-2 split'),
    (4, 3, '4-3 split'),
]
x_pos = np.arange(len(scenarios))
trick_rewards = []
end_game_rewards = []
total_rewards = []
labels = []

for agent_tricks, partner_tricks, label in scenarios:
    trick, end_game, total = calculate_total_game_reward(agent_tricks, partner_tricks)
    trick_rewards.append(trick)
    end_game_rewards.append(end_game)
    total_rewards.append(total)
    labels.append(label)

ax.bar(x_pos, trick_rewards, label='Trick-level Rewards', alpha=0.7, color='skyblue')
ax.bar(x_pos, end_game_rewards, bottom=trick_rewards, label='End-game Bonus', alpha=0.7, color='lightgreen')
ax.plot(x_pos, total_rewards, 'ro-', linewidth=2, markersize=8, label='Total Reward')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Reward', fontsize=11)
ax.set_title('Agent Carries (Team Wins, Agent Contributes More)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/reward_visualizations/reward_scenarios_breakdown.png', dpi=300, bbox_inches='tight')
print("✓ Created: docs/reward_visualizations/reward_scenarios_breakdown.png")
plt.close()

# ============================================================================
# Figure 2: Heatmap of Total Rewards
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 10))

# Create a grid of agent vs partner tricks
agent_tricks_grid = np.arange(0, MAX_TRICKS + 1)
partner_tricks_grid = np.arange(0, MAX_TRICKS + 1)
reward_matrix = np.zeros((len(partner_tricks_grid), len(agent_tricks_grid)))

for i, partner_tricks in enumerate(partner_tricks_grid):
    for j, agent_tricks in enumerate(agent_tricks_grid):
        if agent_tricks + partner_tricks <= MAX_TRICKS:
            _, _, total = calculate_total_game_reward(agent_tricks, partner_tricks)
            reward_matrix[i, j] = total
        else:
            reward_matrix[i, j] = np.nan  # Invalid combinations

# Create heatmap
im = ax.imshow(reward_matrix, cmap='RdYlGn', aspect='auto', origin='lower', vmin=-15, vmax=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Total Reward', fontsize=12, fontweight='bold')

# Add winning line
winning_line_x = []
winning_line_y = []
for agent_tricks in agent_tricks_grid:
    partner_tricks_needed = TEAM_WIN_THRESHOLD - agent_tricks
    if 0 <= partner_tricks_needed <= MAX_TRICKS:
        winning_line_x.append(agent_tricks)
        winning_line_y.append(partner_tricks_needed)

ax.plot(winning_line_x, winning_line_y, 'b--', linewidth=3, label=f'Team Win Threshold ({TEAM_WIN_THRESHOLD} tricks)')

# Labels and annotations
ax.set_xlabel('Agent Tricks Won', fontsize=12, fontweight='bold')
ax.set_ylabel('Partner Tricks Won', fontsize=12, fontweight='bold')
ax.set_title('Total Reward Heatmap: Agent vs Partner Contribution', fontsize=14, fontweight='bold')
ax.set_xticks(agent_tricks_grid)
ax.set_yticks(partner_tricks_grid)
ax.legend(loc='upper right', fontsize=10)

# Add text annotations for key scenarios
key_points = [
    (0, 7, 'Partner\nCarries'),
    (7, 0, 'Agent\nCarries'),
    (3, 4, 'Balanced\nWin'),
    (6, 6, 'Dominant\nWin'),
    (3, 3, 'Team\nLoses'),
]

for agent_t, partner_t, label in key_points:
    if agent_t + partner_t <= MAX_TRICKS:
        _, _, reward = calculate_total_game_reward(agent_t, partner_t)
        ax.annotate(label, xy=(agent_t, partner_t), 
                   xytext=(agent_t + 0.5, partner_t + 0.5),
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('docs/reward_visualizations/reward_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Created: docs/reward_visualizations/reward_heatmap.png")
plt.close()

# ============================================================================
# Figure 3: Win vs Loss Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Impact of Winning vs Losing the Game', fontsize=16, fontweight='bold')

# Left: Team Wins
ax = axes[0]
agent_tricks_range = range(0, 8)
win_rewards = []
win_trick = []
win_endgame = []

for agent_tricks in agent_tricks_range:
    partner_tricks = 7 - agent_tricks
    if partner_tricks < 0:
        continue
    trick, end_game, total = calculate_total_game_reward(agent_tricks, partner_tricks)
    win_rewards.append(total)
    win_trick.append(trick)
    win_endgame.append(end_game)

width = 0.35
x = np.arange(len(agent_tricks_range))

bars1 = ax.bar(x - width/2, win_trick, width, label='Trick Rewards', alpha=0.8, color='skyblue')
bars2 = ax.bar(x - width/2, win_endgame, width, bottom=win_trick, label='End-game +2 Bonus', alpha=0.8, color='lightgreen')
bars3 = ax.bar(x + width/2, win_rewards, width, label='Total', alpha=0.8, color='green')

ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Agent Tricks (Partner gets rest to make 7)', fontsize=11)
ax.set_ylabel('Reward', fontsize=11)
ax.set_title('Team WINS (7 tricks total)', fontweight='bold', color='green')
ax.set_xticks(x)
ax.set_xticklabels(agent_tricks_range)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Right: Team Loses  
ax = axes[1]
agent_tricks_range = range(0, 7)
lose_rewards = []
lose_trick = []
lose_endgame = []

for agent_tricks in agent_tricks_range:
    partner_tricks = 6 - agent_tricks
    if partner_tricks < 0:
        continue
    trick, end_game, total = calculate_total_game_reward(agent_tricks, partner_tricks)
    lose_rewards.append(total)
    lose_trick.append(trick)
    lose_endgame.append(end_game)

x = np.arange(len(agent_tricks_range))

bars1 = ax.bar(x - width/2, lose_trick, width, label='Trick Rewards', alpha=0.8, color='salmon')
bars2 = ax.bar(x - width/2, lose_endgame, width, bottom=lose_trick, label='End-game Penalty', alpha=0.8, color='lightcoral')
bars3 = ax.bar(x + width/2, lose_rewards, width, label='Total', alpha=0.8, color='red')

ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Agent Tricks (Partner gets rest to make 6)', fontsize=11)
ax.set_ylabel('Reward', fontsize=11)
ax.set_title('Team LOSES (6 tricks total)', fontweight='bold', color='red')
ax.set_xticks(x)
ax.set_xticklabels(agent_tricks_range)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('docs/reward_visualizations/win_vs_loss_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Created: docs/reward_visualizations/win_vs_loss_comparison.png")
plt.close()

# ============================================================================
# Figure 4: Per-Trick Reward Timeline
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Simulate a game where agent wins 4, partner wins 3, opponents win 6
tricks = [
    ('Agent', 1.0),
    ('Opponent', -1.0),
    ('Partner', 0.8),
    ('Opponent', -1.0),
    ('Agent', 1.0),
    ('Agent', 1.0),
    ('Opponent', -1.0),
    ('Partner', 0.8),
    ('Opponent', -1.0),
    ('Partner', 0.8),
    ('Agent', 1.0),
    ('Opponent', -1.0),
    ('Opponent', -1.0),
]

cumulative_reward = []
current = 0
colors = []
for winner, reward in tricks:
    current += reward
    cumulative_reward.append(current)
    if winner == 'Agent':
        colors.append('green')
    elif winner == 'Partner':
        colors.append('lightgreen')
    else:
        colors.append('red')

trick_numbers = list(range(1, len(tricks) + 1))

# Plot bars for each trick
ax.bar(trick_numbers, [r for _, r in tricks], color=colors, alpha=0.6, edgecolor='black')

# Plot cumulative line
ax2 = ax.twinx()
ax2.plot(trick_numbers, cumulative_reward, 'bo-', linewidth=2, markersize=8, label='Cumulative Reward')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Trick Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Trick Reward', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
ax.set_title('Per-Trick Reward Timeline Example\n(Agent: 4 tricks, Partner: 3 tricks, Opponents: 6 tricks)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(trick_numbers)
ax.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.6, label='Agent Wins (+1.0)'),
    Patch(facecolor='lightgreen', alpha=0.6, label='Partner Wins (+0.8)'),
    Patch(facecolor='red', alpha=0.6, label='Opponent Wins (-1.0)'),
]
ax.legend(handles=legend_elements, loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('docs/reward_visualizations/per_trick_timeline.png', dpi=300, bbox_inches='tight')
print("✓ Created: docs/reward_visualizations/per_trick_timeline.png")
plt.close()

print("\n✅ All reward visualization graphs created successfully!")
print("📁 Location: docs/reward_visualizations/")
print("\nGenerated files:")
print("  1. reward_scenarios_breakdown.png - Shows 4 different scenarios")
print("  2. reward_heatmap.png - Heatmap of all agent/partner combinations")
print("  3. win_vs_loss_comparison.png - Direct comparison of winning vs losing")
print("  4. per_trick_timeline.png - Example game timeline with cumulative rewards")
