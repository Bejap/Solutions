# Feature Implementation Guide

This document provides detailed descriptions of all newly implemented features, their architecture, usage, and configuration options.

---

## Table of Contents

1. [Dueling DQN Architecture](#1-dueling-dqn-architecture)
2. [N-Step Returns (Multi-Step Learning)](#2-n-step-returns-multi-step-learning)
3. [Learning Rate Scheduling](#3-learning-rate-scheduling)
4. [Epsilon Decay Scheduling](#4-epsilon-decay-scheduling)
5. [Replay Memory Improvements](#5-replay-memory-improvements)
6. [Double DQN](#6-double-dqn)
7. [Early Stopping](#7-early-stopping)
8. [Modular Reward System Framework](#8-modular-reward-system-framework)
9. [Loading and Continue Training Utilities](#9-loading-and-continue-training-utilities)
10. [Reward System Comparison Framework](#10-reward-system-comparison-framework)
11. [Trump Penalty System](#11-trump-penalty-system)

---

## 1. Dueling DQN Architecture

### Overview

Dueling DQN separates the Q-value estimation into two distinct streams:
- **Value Stream V(s)**: Estimates how good it is to be in a state
- **Advantage Stream A(s,a)**: Estimates the relative advantage of each action

The final Q-value is computed as:
```
Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
```

### Why Use Dueling DQN?

- **Better state evaluation**: The value stream learns which states are valuable regardless of actions
- **Improved learning efficiency**: Particularly useful when actions don't always affect the outcome
- **Faster convergence**: Helps distinguish state importance from action importance

### Architecture Diagram

```
Input Features
      │
      ▼
┌─────────────────┐
│  Shared Layers  │
│  (Dense 128→64) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────────┐
│ Value │ │ Advantage │
│ V(s)  │ │ A(s,a)    │
│ (1)   │ │ (13)      │
└───┬───┘ └─────┬─────┘
    │           │
    └─────┬─────┘
          ▼
    Q(s,a) = V(s) + (A(s,a) - mean(A))
```

### Implementation

Located in `Whist/agents/advanced_dqn.py`:

```python
from Whist.agents.advanced_dqn import DuelingDQNAgent

# Create agent
agent = DuelingDQNAgent(
    input_size=372,
    gamma=0.99,
    agent_id=0,
    use_double_dqn=True,  # Combine with Double DQN
    use_n_step=True,      # Combine with n-step returns
    use_lr_scheduling=True # Enable LR scheduling
)

# Training loop
agent.update_replay_memory(transition)
agent.train(terminal_state=done, step=step_count)
```

### Configuration Constants

```python
# In constants.py
HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 64
HIDDEN_LAYER_3_SIZE = 32
DROPOUT_RATE = 0.35
```

---

## 2. N-Step Returns (Multi-Step Learning)

### Overview

N-step returns use rewards from N consecutive steps to compute the target Q-value, providing a balance between 1-step TD (biased but low variance) and Monte Carlo (unbiased but high variance).

### Formula

Standard TD (1-step):
```
target = r₁ + γ * max(Q(s'))
```

N-step TD:
```
target = r₁ + γ*r₂ + γ²*r₃ + ... + γⁿ⁻¹*rₙ + γⁿ * max(Q(sₙ'))
```

### Benefits

- **Faster credit assignment**: Rewards propagate back N steps immediately
- **Better bias-variance trade-off**: Reduces bootstrapping bias
- **Improved learning speed**: Especially effective for delayed rewards

### Implementation

Located in `Whist/agents/advanced_dqn.py`:

```python
class NStepReplayBuffer:
    """
    N-step replay buffer that computes n-step returns automatically.
    """
    def __init__(self, maxlen=50000, n_steps=3, gamma=0.99):
        self.buffer = deque(maxlen=maxlen)
        self.n_step_buffer = deque(maxlen=n_steps)
        # ...

    def append(self, transition):
        # Automatically computes n-step returns when buffer has enough transitions
        # Handles terminal states by flushing remaining transitions
```

### Configuration

```python
# In constants.py
N_STEP_RETURNS = 3  # Number of steps (1 = standard TD)

# Usage with DuelingDQNAgent
agent = DuelingDQNAgent(
    input_size=372,
    use_n_step=True  # Enable n-step returns
)
```

### Effect on Discount Factor

When using n-step returns, the effective discount for the bootstrap value becomes:
```python
effective_gamma = gamma ** N_STEP_RETURNS  # e.g., 0.99³ ≈ 0.97
```

---

## 3. Learning Rate Scheduling

### Overview

Learning rate scheduling gradually reduces the learning rate during training, allowing:
- **Fast initial learning**: High LR for quick early progress
- **Fine-tuned convergence**: Low LR for stable final learning
- **Prevented overshooting**: Avoids oscillation near optimal values

### Exponential Decay Schedule

```
lr(step) = initial_lr * decay_rate^(step / decay_steps)
```

### Implementation

```python
# Automatically created in DuelingDQNAgent
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LEARNING_RATE,
    decay_steps=LR_DECAY_STEPS,
    decay_rate=LR_DECAY_RATE,
    staircase=True  # Discrete steps (not continuous)
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Configuration

```python
# In constants.py
INITIAL_LEARNING_RATE = 0.001  # Starting learning rate
LR_DECAY_STEPS = 1000          # Decay every N training steps
LR_DECAY_RATE = 0.96           # Multiply by this factor each period
MIN_LEARNING_RATE = 0.0001     # Floor value
```

### Monitoring Current LR

```python
agent = DuelingDQNAgent(...)
current_lr = agent.get_current_lr()
print(f"Current learning rate: {current_lr}")
```

---

## 4. Epsilon Decay Scheduling

### Overview

The `EpsilonScheduler` class provides flexible exploration strategies beyond simple exponential decay.

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `exponential` | Standard exponential decay | General use |
| `linear` | Linear decay to minimum | Consistent exploration reduction |
| `step` | Discrete jumps at specified episodes | Controlled phases |
| `cosine` | Cosine annealing with warm restarts | Cyclical exploration |

### Implementation

Located in `Whist/agents/advanced_dqn.py`:

```python
from Whist.agents.advanced_dqn import EpsilonScheduler

# Exponential decay (default)
scheduler = EpsilonScheduler(
    initial_epsilon=1.0,
    min_epsilon=0.01,
    decay_type='exponential',
    decay_rate=0.995
)

# Linear decay
scheduler = EpsilonScheduler(
    decay_type='linear',
    total_episodes=1000
)

# Step decay (discrete phases)
scheduler = EpsilonScheduler(
    decay_type='step',
    step_episodes=[300, 600, 900],
    step_values=[0.5, 0.2, 0.05]
)

# Cosine annealing with warm restarts
scheduler = EpsilonScheduler(
    decay_type='cosine',
    warm_restart_period=200  # Restart every 200 episodes
)

# Usage in training loop
for episode in range(num_episodes):
    epsilon = scheduler.get_epsilon(episode)
    # ... training code ...
    scheduler.step()  # Advance to next episode
```

### Cosine Annealing Visualization

```
ε  1.0 │\    /\    /\
       │ \  /  \  /  \
  0.5  │  \/    \/    \
       │                \
  0.01 │─────────────────────
       └────────────────────→ episodes
         0   200  400  600
```

### Configuration

```python
# In constants.py
EPSILON_DECAY_TYPE = 'exponential'
EPSILON_STEP_DECAY_EPISODES = [300, 600, 900]
EPSILON_STEP_DECAY_VALUES = [0.5, 0.2, 0.05]
```

---

## 5. Replay Memory Improvements

### Overview

The replay memory configuration has been significantly increased for better training stability and sample diversity.

### Changes

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `REPLAY_MEMORY_SIZE` | 100 | 50,000 | Much larger experience buffer |
| `MIN_REPLAY_MEMORY_SIZE` | 100 | 1,000 | More samples before training starts |
| `MINIBATCH_SIZE` | 32 | 64 | Larger batches for stable gradients |

### Benefits

- **More diverse samples**: Larger buffer contains wider variety of experiences
- **Reduced correlation**: Better decorrelation between consecutive samples
- **Stable training**: Larger minibatches provide more stable gradient estimates

### Configuration

```python
# In constants.py
REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
```

---

## 6. Double DQN

### Overview

Double DQN addresses the overestimation bias in standard DQN by decoupling action selection from value estimation.

### The Problem with Standard DQN

Standard DQN uses `max(Q_target(s'))` which tends to overestimate values because the same network both selects and evaluates the best action.

### Double DQN Solution

1. **Online network** selects the best action
2. **Target network** evaluates that action's value

```python
# Standard DQN (overestimates)
max_future_q = np.max(target_model.predict(next_state))

# Double DQN (more accurate)
best_action = np.argmax(online_model.predict(next_state))
max_future_q = target_model.predict(next_state)[best_action]
```

### Implementation

Available in all DQN agents:

```python
from Whist.agents.simple_whist_DQN import DQNAgent

agent = DQNAgent(
    input_size=372,
    gamma=0.99,
    use_double_dqn=True  # Enabled by default
)
```

---

## 7. Early Stopping

### Overview

Early stopping monitors training progress and stops when performance plateaus, preventing:
- **Overfitting**: Stopping before the model starts memorizing
- **Wasted compute**: No benefit from continued training after convergence
- **Degradation**: Prevents performance regression

### Implementation

Located in `Whist/training/training_logic.py`:

```python
trainer = WhistTrainer(
    num_games=5000,
    early_stopping_patience=100,      # Stop after N episodes without improvement
    early_stopping_min_delta=0.01     # Minimum improvement threshold
)
```

### How It Works

1. Monitors rolling average reward over last N episodes
2. If no improvement of at least `min_delta` for `patience` episodes, training stops
3. Saves the best model before stopping

### Configuration

```python
# In WhistTrainer constructor
early_stopping_patience=100   # Episodes to wait for improvement
early_stopping_min_delta=0.01 # Minimum reward improvement required
```

---

## 8. Modular Reward System Framework

### Overview

The reward system framework provides a flexible, extensible architecture for implementing and comparing different reward strategies.

### Available Reward Systems

| System | Description | Rating |
|--------|-------------|--------|
| `current` | Original fixed reward system | 7.5/10 |
| `hybrid_custom` | Recommended approach with curriculum scaling | 9/10 |
| `sparse` | Only game-end rewards | 4/10 |
| `hierarchical` | Multi-level rewards with scaling | 6.5/10 |
| `curriculum` | Transitions from dense to sparse | 8.5/10 |

### Usage

Located in `Whist/training/reward_systems.py`:

```python
from Whist.training.reward_systems import get_reward_system

# Create a reward system
reward_system = get_reward_system('hybrid_custom', curriculum_threshold=3000)

# Calculate reward
reward = reward_system.calculate_reward(
    game_state={'trump_penalty': -5.0, 'team_tricks': 4},
    action=5,
    outcome='agent_wins',
    episode=1000
)

# End-game reward
end_reward = reward_system.calculate_end_game_reward(
    team_won=True,
    agent_tricks=8,
    partner_tricks=5,
    total_tricks=13
)
```

### Hybrid Reward System (Recommended)

The `HybridRewardSystem` implements Approach #11 with:

1. **Adjusted trick rewards**:
   - Agent wins: +1.0
   - Partner wins: +0.6 (reduced from 0.8)
   - Opponent wins: -0.8 (reduced from -1.0)

2. **Curriculum-based penalty scaling**:
   - Episodes < threshold: 100% penalties (full learning signal)
   - Episodes ≥ threshold: 30% penalties (reduced to prevent dominance)

3. **Simplified end-game**:
   - Win: +5.0
   - Loss: -5.0

4. **Optional win probability estimation**:
   - Provides dense feedback based on estimated win probability changes

### Creating Custom Reward Systems

```python
from Whist.training.reward_systems import BaseRewardSystem

class MyRewardSystem(BaseRewardSystem):
    def __init__(self):
        super().__init__("my_system")
    
    def calculate_reward(self, game_state, action, outcome, episode):
        # Your reward logic here
        return reward
    
    def calculate_end_game_reward(self, team_won, agent_tricks, 
                                  partner_tricks, total_tricks):
        # Your end-game logic here
        return reward
```

---

## 9. Loading and Continue Training Utilities

### Overview

Utilities for loading pre-trained agents and continuing training from checkpoints, enabling:
- **Incremental training**: Train in phases
- **Transfer learning**: Start from pre-trained weights
- **Experiment continuation**: Resume interrupted training

### Standard DQN Agent Loading

Located in `Whist/training/loading_and_continue_training.py`:

```python
from Whist.training.loading_and_continue_training import (
    load_agent_from_weights,
    load_agent_from_keras,
    ContinueTraining
)

# Load from weights file
agent = load_agent_from_weights(
    weights_path="Weights/agent_player_0_ep1000.weights.h5",
    input_size=372,
    gamma=0.99,
    agent_id=0,
    use_double_dqn=True
)

# Load from full Keras model
agent = load_agent_from_keras(
    model_path="Models/full_agent_player_0_ep1000.keras",
    gamma=0.99,
    agent_id=0
)

# Continue training with ContinueTraining class
continue_trainer = ContinueTraining(
    agent_0_path="Weights/agent_player_0_ep1000.weights.h5",
    agent_2_path="Weights/agent_player_2_ep1000.weights.h5",
    path_type='weights',  # or 'keras'
    starting_episode=1000,
    num_additional_games=2000,
    epsilon=0.3,  # Lower epsilon for pre-trained agents
    use_double_dqn=True,
    early_stopping_patience=150
)

trainer = continue_trainer.continue_training()
trainer.plot_results()
```

### Embedded DQN Agent Loading

Located in `Whist/training/loading_and_continue_training_embedded.py`:

```python
from Whist.training.loading_and_continue_training_embedded import (
    load_embedded_agent_from_weights,
    load_embedded_agent_from_keras,
    ContinueEmbeddedTraining
)

# Load embedded agent
agent = load_embedded_agent_from_weights(
    weights_path="Weights/embedded_agent_player_0_ep1000.weights.h5",
    embedding_dim=8,
    gamma=0.99,
    agent_id=0
)

# Continue embedded training
continue_trainer = ContinueEmbeddedTraining(
    agent_0_path="Weights/embedded_agent_player_0_ep1000.weights.h5",
    agent_2_path="Weights/embedded_agent_player_2_ep1000.weights.h5",
    path_type='weights',
    embedding_dim=8,
    starting_episode=1000,
    num_additional_games=2000,
    epsilon=0.3
)

trainer = continue_trainer.continue_training()
```

---

## 10. Reward System Comparison Framework

### Overview

A framework for empirically comparing different reward systems to identify the best approach for your specific use case.

### Usage

Located in `Whist/training/reward_comparison.py`:

```python
from Whist.training.reward_comparison import RewardSystemComparator

# Compare multiple reward systems
comparator = RewardSystemComparator(
    reward_systems=['current', 'hybrid_custom', 'sparse', 'curriculum'],
    n_training_episodes=5000,
    n_eval_games=100
)

# Run comparison
results = comparator.compare_all()

# Get best system
best_name, metrics = comparator.get_best_system(results, metric='win_rate')

# Print summary
comparator.print_summary(results)
```

### Metrics Collected

| Metric | Description |
|--------|-------------|
| `win_rate` | Percentage of games won in evaluation |
| `avg_tricks` | Average tricks won per game |
| `training_time` | Total training time in seconds |
| `convergence_speed` | Episodes to reach performance threshold |

### Output Example

```
╔════════════════════════════════════════════════════════════╗
║              Reward System Comparison Results              ║
╠══════════════════╦══════════╦═══════════╦═════════════════╣
║ System           ║ Win Rate ║ Avg Tricks║ Convergence     ║
╠══════════════════╬══════════╬═══════════╬═════════════════╣
║ hybrid_custom    ║   68.5%  ║    7.2    ║    1200 eps     ║
║ curriculum       ║   65.0%  ║    6.9    ║    1500 eps     ║
║ current          ║   62.0%  ║    6.7    ║    1800 eps     ║
║ sparse           ║   45.0%  ║    5.8    ║    3500 eps     ║
╚══════════════════╩══════════╩═══════════╩═════════════════╝
```

---

## 11. Trump Penalty System

### Overview

Penalties for suboptimal trump usage to teach strategic card play:

1. **TRUMP_NOT_USED_PENALTY**: Not using trump when opponent is winning and you have trump (partner not already winning)
2. **TRUMP_OVERPLAY_PENALTY**: Using unnecessarily high trump when a lower trump would win
3. **PARTNER_OVERPLAY_PENALTY**: Taking the trick from a partner who is already winning

### Implementation

Located in `Whist/core/whist.py`:

```python
def calculate_trump_penalty(self, current_player, played_card, hand_before_play):
    """
    Calculate penalty for suboptimal trump play.
    
    Returns:
        float: Penalty value (negative) or 0.0 if no penalty
    """
    # ... penalty logic ...
```

### Configuration

```python
# In constants.py
TRUMP_NOT_USED_PENALTY = -7.0   # Penalty for not trumping when should
TRUMP_OVERPLAY_PENALTY = -5.0   # Penalty for playing unnecessarily high trump
PARTNER_OVERPLAY_PENALTY = -8.0 # Penalty for taking from winning partner
```

### Example Scenarios

**Scenario 1: Not Using Trump**
- Opponent plays Q♥ (currently winning)
- Agent has no hearts but has 2♠ (trump)
- Agent plays 7♦ (off-suit, loses)
- Penalty: `TRUMP_NOT_USED_PENALTY`

**Scenario 2: Trump Overplay**
- Opponent plays 5♥
- Agent has no hearts but has 2♠, K♠ (both trump)
- Agent plays K♠ when 2♠ would win
- Penalty: `TRUMP_OVERPLAY_PENALTY`

**Scenario 3: Partner Overplay**
- Partner (N) plays K♠ (currently winning)
- Opponent plays 5♠
- Agent has A♠ and 3♠
- Agent plays A♠ (unnecessary, wastes high card)
- Penalty: `PARTNER_OVERPLAY_PENALTY`

---

## Quick Reference: Enabling All Features

```python
from Whist.agents.advanced_dqn import DuelingDQNAgent, EpsilonScheduler
from Whist.training.reward_systems import get_reward_system
from Whist.training.training_logic import WhistTrainer

# Create agent with all features
agent = DuelingDQNAgent(
    input_size=372,
    gamma=0.99,
    use_double_dqn=True,      # Double DQN
    use_n_step=True,          # N-step returns
    use_lr_scheduling=True    # Learning rate scheduling
)

# Epsilon scheduler with cosine annealing
epsilon_scheduler = EpsilonScheduler(
    decay_type='cosine',
    warm_restart_period=200
)

# Hybrid reward system
reward_system = get_reward_system('hybrid_custom', curriculum_threshold=3000)

# Training with early stopping
trainer = WhistTrainer(
    num_games=5000,
    use_dueling_dqn=True,
    epsilon_decay_type='cosine',
    early_stopping_patience=100
)
```

---

## Related Documentation

- [Optimization Suggestions](optimization_suggestions.md) - Full list of implemented and pending optimizations
- [Reward Suggestions](reward_suggestions.md) - Alternative reward system approaches
- [Constants Guide](CONSTANTS_GUIDE.md) - All configuration constants
- [Architecture Guide](architecture.md) - System architecture overview
