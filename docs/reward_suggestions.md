# Alternative Reward System Approaches

This document explores alternative reward system designs for the Whist DQN training, provides a critical evaluation of the current approach, and rates different strategies.

---

## Current Reward System: Analysis & Critique

### Overview
The current reward system uses a **sparse + shaped hybrid approach** with multiple reward components:

**Per-Trick Rewards** (immediate):
- Agent wins trick: `+1.0`
- Partner wins trick: `+0.9`
- Opponent wins trick: `-1.1`

**Trump Penalties** (immediate):
- Not using trump when optimal: `TRUMP_NOT_USED_PENALTY`
- Using unnecessarily high trump: `TRUMP_OVERPLAY_PENALTY`
- Taking trick from winning partner: `PARTNER_OVERPLAY_PENALTY`

**Per-Card Rewards** (optional, immediate):
- Matching EW strategy: `PER_CARD_EW_STRATEGY_REWARD`

**End-Game Rewards** (terminal):
- Formula: `((13 + (tricks_won - 13)) / 10) ** (1 + (tricks_won / 13)) + team_bonus`
- Team bonus: +2 if team wins (≥7 tricks), 0 otherwise

### Rating: 7.5/10

**Strengths:**
1. ✅ **Clear immediate feedback**: Per-trick rewards provide quick learning signals
2. ✅ **Partner coordination**: Rewards partner tricks (0.9) encourages teamwork
3. ✅ **Strategic guidance**: Trump penalties teach optimal trump usage
4. ✅ **Balanced**: Combines immediate and delayed rewards
5. ✅ **Configurable**: Many tunable parameters for different training strategies

**Weaknesses & Flaws:**
1. ❌ **Potential credit assignment issues**: Multiple reward sources can create conflicting signals
   - Example: Agent might get +1 for winning a trick but -5 for trump overplay, net negative despite "winning"
2. ❌ **Reward scaling inconsistency**: Trump penalties (-5 to -8) dwarf per-trick rewards (±1), potentially overwhelming the primary learning signal
3. ❌ **Dense reward complexity**: Many reward components may slow convergence as agent tries to optimize multiple objectives simultaneously
4. ❌ **EW imitation risk**: Per-card rewards for matching EW strategy could limit exploration and prevent discovering superior strategies
5. ❌ **No explicit communication penalty**: Agents might develop strategies that work but don't generalize well to team play

### Key Concerns:
- **Magnitude imbalance**: A single trump penalty can erase 5-8 trick wins worth of positive reward
- **Multi-objective optimization**: Agents must balance trick-winning, trump conservation, partner coordination, and EW strategy matching
- **Exploration inhibition**: Strong negative penalties might discourage exploration of novel trump usage patterns

---

## Alternative Approach 1: Pure Sparse Rewards

### Description
Only reward at game end based on final team performance. No intermediate rewards, penalties, or shaping.

### Implementation
```python
# During game: all rewards = 0
reward = [0, 0, 0, 0]

# At game end:
team_1_tricks = agent_0_tricks + agent_2_tricks
if team_1_tricks >= winning_threshold:
    reward[0] = +1
    reward[2] = +1
else:
    reward[0] = -1
    reward[2] = -1
```

### Rating: 4/10

**Advantages:**
- ✅ Clear optimization objective: win the game
- ✅ No credit assignment confusion
- ✅ Maximum exploration freedom
- ✅ Simplest to implement and debug

**Disadvantages:**
- ❌ Very slow learning: Sparse reward problem is severe
- ❌ No guidance on HOW to win
- ❌ Requires massive experience collection
- ❌ May never learn trump strategy without shaping

**Best for:** Environments with very short episodes or when you have unlimited training time

---

## Alternative Approach 2: Hierarchical Rewards

### Description
Decompose the task into a hierarchy: card selection → trick winning → game winning. Use separate reward functions at each level.

### Implementation
```python
# Level 1: Card quality (immediate)
card_rank_reward = normalize_rank(card_played) * 0.1

# Level 2: Trick outcome (per trick)
if won_trick:
    trick_reward = +1.0
elif partner_won_trick:
    trick_reward = +0.5  # Reduced to emphasize self-winning
else:
    trick_reward = -0.5  # Less severe than current -1.0

# Level 3: Game outcome (terminal)
if team_won:
    game_reward = +10.0  # Dominant signal
else:
    game_reward = -10.0

# Total reward at each step
reward = card_rank_reward + trick_reward + game_reward
```

### Rating: 6.5/10

**Advantages:**
- ✅ Clear hierarchy reduces confusion
- ✅ Strong terminal signal emphasizes winning
- ✅ Intermediate rewards still guide learning
- ✅ Reduced penalty magnitude

**Disadvantages:**
- ❌ Card rank reward may be misleading (high card isn't always best)
- ❌ Still requires tuning multiple reward scales
- ❌ Doesn't explicitly teach trump strategy
- ❌ Partner coordination less emphasized

**Best for:** When you want clearer optimization hierarchy and stronger end-game focus

---

## Alternative Approach 3: Potential-Based Reward Shaping

### Description
Use potential-based shaping to provide dense rewards while preserving optimal policy. Potential function measures "closeness to winning."

### Implementation
```python
def potential(state):
    """Estimate value of current game state"""
    team_tricks = agent_0_tricks + agent_2_tricks
    tricks_remaining = max_tricks - total_tricks_played
    
    # Can we still win?
    max_possible = team_tricks + tricks_remaining
    if max_possible < winning_threshold:
        return -10.0  # Can't win anymore
    
    # How close to winning?
    tricks_needed = winning_threshold - team_tricks
    return (tricks_remaining - tricks_needed) * 0.5

# Shaped reward
reward = actual_reward + gamma * potential(next_state) - potential(current_state)
```

### Rating: 8/10

**Advantages:**
- ✅ Theoretically sound: Preserves optimal policy (Ng et al., 1999)
- ✅ Dense rewards without bias
- ✅ Guides agent toward winning states
- ✅ Automatically adjusts based on game progress
- ✅ No hand-tuned penalty magnitudes needed

**Disadvantages:**
- ❌ Requires good potential function design
- ❌ Doesn't explicitly teach trump strategy (but agents can discover it)
- ❌ More complex to implement and debug
- ❌ Potential function needs domain knowledge

**Best for:** When you want principled dense rewards without manual tuning

---

## Alternative Approach 4: Curiosity-Driven Intrinsic Rewards

### Description
Add intrinsic motivation bonuses for exploring novel states or actions, combined with sparse game-winning rewards.

### Implementation
```python
# Intrinsic Curiosity Module (ICM)
def intrinsic_reward(state, action, next_state):
    # Predict next state from current state + action
    predicted_next = forward_model(state, action)
    
    # Reward for surprise (prediction error)
    prediction_error = MSE(predicted_next, next_state)
    return beta * prediction_error

# Total reward
reward = extrinsic_reward + intrinsic_reward
```

### Rating: 6/10

**Advantages:**
- ✅ Encourages exploration naturally
- ✅ May discover creative trump strategies
- ✅ Works with sparse extrinsic rewards
- ✅ Self-supervised learning component

**Disadvantages:**
- ❌ Adds significant complexity (need forward model)
- ❌ Can be distracted by irrelevant novelty
- ❌ Harder to train and debug
- ❌ May explore too much, not exploit enough
- ❌ Computational overhead

**Best for:** When exploration is critical and you have computational resources

---

## Alternative Approach 5: Opponent Modeling Rewards

### Description
Reward agents for accurately predicting opponent behavior and exploiting weaknesses.

### Implementation
```python
# Predict opponent's next move
opponent_prediction = opponent_model(game_state)

# Reward for accurate prediction
if opponent_played == predicted_card:
    prediction_reward = +0.2

# Reward for exploiting prediction
if we_won_trick and we_used_prediction:
    exploitation_reward = +0.3

# Combine with basic trick rewards
total_reward = trick_reward + prediction_reward + exploitation_reward
```

### Rating: 7/10

**Advantages:**
- ✅ Encourages strategic thinking
- ✅ Teaches opponent modeling explicitly
- ✅ Can adapt to different opponent strategies
- ✅ Adds competitive intelligence

**Disadvantages:**
- ❌ Requires opponent model training
- ❌ May overfit to training opponents
- ❌ Doesn't work well with random opponents
- ❌ Additional complexity

**Best for:** Competitive settings with consistent opponents

---

## Alternative Approach 6: Advantage-Based Rewards

### Description
Reward based on how much better/worse the action was compared to average expected value.

### Implementation
```python
# Estimate value of all legal actions
q_values = []
for action in valid_actions:
    q_values.append(Q_network(state, action))

# Average value
baseline = mean(q_values)

# Actual outcome value
actual_value = reward_from_environment + gamma * V(next_state)

# Advantage-based reward
advantage = actual_value - baseline
shaped_reward = sign(advantage) * min(abs(advantage), max_reward)
```

### Rating: 7.5/10

**Advantages:**
- ✅ Normalizes rewards across different game states
- ✅ Reduces variance in learning
- ✅ Focuses on relative improvement
- ✅ Compatible with actor-critic architectures

**Disadvantages:**
- ❌ Requires value function estimation
- ❌ Can be unstable early in training
- ❌ More complex implementation
- ❌ Baseline estimation adds computation

**Best for:** Actor-critic or policy gradient methods

---

## Alternative Approach 7: Curriculum-Based Reward Scheduling

### Description
Start with dense shaped rewards, gradually transition to sparse rewards as agent improves.

### Implementation
```python
def get_reward(episode_number, max_episodes):
    # Shaping weight decreases over time
    alpha = max(0, 1 - episode_number / (max_episodes * 0.7))
    
    # Dense shaped component
    shaped_reward = calculate_shaped_reward(state, action)
    
    # Sparse true reward
    true_reward = calculate_sparse_reward(final_outcome)
    
    # Blend rewards
    return alpha * shaped_reward + (1 - alpha) * true_reward
```

### Rating: 8.5/10

**Advantages:**
- ✅ Best of both worlds: fast early learning + optimal final policy
- ✅ Gradual transition reduces destabilization
- ✅ Can use aggressive shaping initially
- ✅ Converges to true objective
- ✅ Proven effective in practice

**Disadvantages:**
- ❌ Requires tuning curriculum schedule
- ❌ Adds hyperparameter complexity
- ❌ Need to define both reward systems
- ❌ Training time may increase

**Best for:** When you want fast initial learning but optimal final performance

---

## Alternative Approach 8: Win Probability Rewards

### Description
Reward based on change in estimated win probability after each action.

### Implementation
```python
# Train a separate win probability estimator
win_prob_before = win_estimator(state_before)
win_prob_after = win_estimator(state_after)

# Reward is change in win probability
reward = (win_prob_after - win_prob_before) * scaling_factor

# Add terminal reward
if game_over:
    if won:
        reward += 1.0
    else:
        reward -= 1.0
```

### Rating: 8/10

**Advantages:**
- ✅ Direct optimization of win probability
- ✅ Dense signal throughout game
- ✅ Automatically accounts for game progress
- ✅ Interpretable rewards
- ✅ Focuses on what matters: winning

**Disadvantages:**
- ❌ Requires win probability estimator
- ❌ Estimator needs to be accurate
- ❌ Chicken-and-egg problem early in training
- ❌ Additional model to train and maintain

**Best for:** When you have or can build a good win predictor

---

## Alternative Approach 9: Imitation + Refinement

### Description
Start with imitation learning from expert strategy, then refine with reinforcement learning.

### Implementation
```python
# Phase 1: Imitation Learning (behavioral cloning)
if episode < imitation_phase_episodes:
    # Minimize difference from expert
    loss = cross_entropy(agent_action_probs, expert_actions)
    reward = 0  # Not used, supervised learning
else:
    # Phase 2: Reinforcement learning refinement
    # Use sparse or shaped rewards
    reward = calculate_rl_reward(state, action, outcome)
```

### Rating: 9/10

**Advantages:**
- ✅ Fast initial learning from expert
- ✅ Good starting policy
- ✅ RL phase can discover improvements
- ✅ Reduces exploration space
- ✅ Proven effective in games

**Disadvantages:**
- ❌ Requires expert demonstrations (EW strategy available)
- ❌ May get stuck in local optima near expert
- ❌ Two-phase training is more complex
- ❌ Risk of degrading expert performance

**Best for:** When expert strategy exists but may not be optimal

---

## Alternative Approach 10: Multi-Agent Reward (Team Focus)

### Description
Reward both agents based on team performance, emphasizing collaboration.

### Implementation
```python
# Shared team reward
team_tricks = agent_0_tricks + agent_2_tricks
team_reward = team_tricks / max_possible_team_tricks

# Individual contribution bonus
if agent_won_trick:
    contribution = +0.2
elif agent_helped_partner:  # Detected heuristically
    contribution = +0.1
else:
    contribution = 0

# Final reward
reward_agent_0 = team_reward + contribution_agent_0
reward_agent_2 = team_reward + contribution_agent_2
```

### Rating: 7/10

**Advantages:**
- ✅ Emphasizes team success
- ✅ Reduces individual competition between partners
- ✅ Natural for cooperative game
- ✅ Encourages coordinated strategies

**Disadvantages:**
- ❌ Credit assignment still challenging
- ❌ Individual contribution hard to measure
- ❌ May not learn who should take tricks
- ❌ Requires careful balance of team vs. individual

**Best for:** Highly cooperative team scenarios

---

## Recommendations

### For Your Current System:

**Keep (Good decisions):**
1. Per-trick rewards structure (±1.0, 0.9, -1.1)
2. Configurable reward system with constants
3. Optional per-card rewards for bootstrapping
4. End-game rewards for final performance

**Consider Changing:**

1. **Reduce trump penalty magnitudes** (High Priority)
   - Current penalties (-5 to -8) may be too strong
   - Recommendation: Scale to -1.0 to -2.0 range to match trick rewards
   - This prevents penalties from dominating the learning signal

2. **Implement reward scaling schedule** (High Priority)
   - Start with stronger penalties to teach quickly
   - Gradually reduce penalty magnitude as training progresses
   - Allows fast learning without long-term distortion

3. **Note on partner reward** (Informational)
   - Current: +0.9 for partner winning
   - The value of 0.9 (vs 1.0 for self) creates a slight preference for winning tricks directly
   - This balance encourages both teamwork and individual initiative

4. **Add reward normalization** (Medium Priority)
   ```python
   # Normalize rewards to consistent scale
   normalized_reward = reward / running_std(all_rewards)
   ```

5. **Consider curriculum approach** (Low Priority)
   - Phase 1: Dense shaping with penalties
   - Phase 2: Reduce shaping, increase sparse rewards
   - Phase 3: Pure performance-based rewards

### Best Alternative for Your Use Case:

**Recommendation: Hybrid Approach #11 (Custom)**

Combine the best elements:

```python
def calculate_reward(state, action, outcome, episode):
    # 1. Base trick reward (unchanged)
    trick_reward = {
        'agent_wins': +1.0,
        'partner_wins': +0.6,  # Reduced from 0.8
        'opponent_wins': -0.8   # Reduced from -1.0
    }[outcome]
    
    # 2. Scaled trump penalties (reduced magnitude)
    if episode < curriculum_threshold:
        penalty_scale = 1.0  # Full penalties
    else:
        penalty_scale = 0.3  # Reduced penalties
    
    trump_penalty = calculate_trump_penalty() * penalty_scale
    
    # 3. Win probability bonus (new)
    win_prob_change = estimate_win_prob_change(state, next_state)
    win_bonus = win_prob_change * 0.5
    
    # 4. End-game reward (simplified)
    if game_over:
        if team_won:
            end_reward = +5.0
        else:
            end_reward = -5.0
    else:
        end_reward = 0
    
    # Combine components
    total_reward = trick_reward + trump_penalty + win_bonus + end_reward
    
    return total_reward
```

**Rating: 9/10**

This approach:
- Maintains your successful trick reward structure
- Reduces penalty magnitude issues
- Adds curriculum for penalty scaling
- Simplifies end-game rewards
- Adds win probability shaping for better learning

---

## Implementation Priority

If implementing changes to current system:

**Phase 1 (Quick Wins):**
1. Reduce trump penalty magnitudes by 60-80%
2. Adjust partner reward from 0.8 to 0.6
3. Add reward normalization/clipping

**Phase 2 (Moderate Effort):**
1. Implement curriculum-based penalty scaling
2. Simplify end-game reward formula
3. Add win probability estimator

**Phase 3 (Advanced):**
1. Experiment with potential-based shaping
2. Implement imitation + refinement pipeline
3. Add multi-agent team rewards

---

## Testing Different Approaches

To compare reward systems empirically:

```python
# Framework for reward system comparison
reward_systems = {
    'current': CurrentRewardSystem(),
    'sparse': SparseRewardSystem(),
    'hierarchical': HierarchicalRewardSystem(),
    'curriculum': CurriculumRewardSystem(),
    # ... etc
}

results = {}
for name, reward_system in reward_systems.items():
    agent = train_agent(reward_system, n_episodes=5000)
    performance = evaluate_agent(agent, n_games=100)
    results[name] = {
        'win_rate': performance.win_rate,
        'avg_tricks': performance.avg_tricks,
        'training_time': performance.training_time,
        'convergence_speed': performance.convergence_speed
    }

# Compare and select best
best_system = max(results.items(), key=lambda x: x[1]['win_rate'])
```

---

## Conclusion

Your current reward system is **solid and well-designed** (7.5/10), but has room for improvement primarily around:
1. Penalty magnitude scaling
2. Reward component balance
3. Learning curriculum

The **main risk** is that strong trump penalties may dominate learning and prevent agents from discovering novel strategies. Consider implementing **curriculum-based penalty scaling** as the highest-priority improvement.

For maximum performance, the **Hybrid Approach #11** (combining your current structure with curriculum scaling and win probability bonuses) is recommended.
