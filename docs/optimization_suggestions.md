# Optimization Suggestions

This document provides light and simple optimization suggestions for the Deep Simple Whist DQN implementation.

**Note**: All constant values are determined by the code in `constants.py`, not by this documentation. This document describes the effects of adjusting constants, not their current values.

## Training Optimizations

### 1. Batch Size Tuning
- **Status**: ✅ Implemented
- **Constant**: `MINIBATCH_SIZE`
- **Effect of increasing**: Better gradient estimates, more stable learning, but slower updates and more memory
- **Effect of decreasing**: Faster updates, less memory, but noisier gradients and less stable learning
- **Trade-off**: Balance between training stability and computational efficiency

### 2. Replay Memory Size
- **Status**: ✅ Implemented
- **Constant**: `REPLAY_MEMORY_SIZE`, `MIN_REPLAY_MEMORY_SIZE`
- **Effect of increasing**: More diverse training experiences, better exploration of past states, but more memory usage
- **Effect of decreasing**: Less memory usage, faster sampling, but less experience diversity
- **Trade-off**: Balance between experience diversity and memory constraints
- **Notes**: Increased from 100 to 50,000 for significantly improved experience diversity

### 3. Learning Rate
- **Status**: ✅ Implemented
- **Configuration**: Adam optimizer with exponential decay scheduling
- **Constants**: `INITIAL_LEARNING_RATE`, `LR_DECAY_STEPS`, `LR_DECAY_RATE`, `MIN_LEARNING_RATE`
- **Effect of increasing initial rate**: Faster initial learning but risk of instability and overshooting optima
- **Effect of decreasing initial rate**: More stable training, better final performance, but slower convergence
- **Implementation**: Exponential decay scheduler that reduces LR over training
- **Notes**: Automatically enabled in DuelingDQNAgent, configurable via `use_lr_scheduling`

## Model Architecture Optimizations

### 4. Dropout Regularization
- **Status**: ✅ Implemented
- **Constant**: `DROPOUT_RATE`
- **Effect of increasing**: Better generalization, less overfitting, but slower learning and potential underfitting
- **Effect of decreasing**: Faster learning, better training performance, but risk of overfitting
- **Trade-off**: Balance between generalization and learning speed

### 5. Activation Functions
- **Current**: ReLU activation
- **Suggestion**: Try LeakyReLU or ELU for better gradient flow
- **Impact**: Alternative activations can reduce dead neuron problem
- **Notes**: ELU provides smoother gradients with negative values

## Reward System Optimizations

### 6. Trump Play Penalties
- **Status**: ✅ Implemented
- **Constants**: `TRUMP_NOT_USED_PENALTY`, `TRUMP_OVERPLAY_PENALTY`, `PARTNER_OVERPLAY_PENALTY`
- **Effect of more negative values**: Stronger punishment for mistakes, faster learning of proper behavior, but higher initial variance
- **Effect of less negative values**: Gentler learning, more exploration of alternatives, but slower convergence to optimal strategy
- **Trade-off**: Balance between learning speed and training stability
- **Impact**: Teaches optimal trump usage and partner coordination

### 7. Per-Card Rewards
- **Status**: ✅ Implemented (Configurable)
- **Constants**: `ENABLE_PER_CARD_REWARD`, `PER_CARD_EW_STRATEGY_REWARD`
- **Effect of increasing reward**: Stronger immediate feedback, faster initial learning, but may cause over-imitation of EW strategy
- **Effect of decreasing reward**: More independent learning, diverse strategies, but slower initial progress
- **Trade-off**: Balance between guided learning and strategic independence
- **Suggestion**: Consider disabling after initial training to encourage independent strategy

### 8. Exploration Phase
- **Status**: ✅ Implemented
- **Constant**: `EXPLORATION_GAMES`
- **Effect of increasing**: More initial experience diversity, better coverage of state space, but delayed actual learning
- **Effect of decreasing**: Faster start to actual training, but potentially poor initial experience distribution
- **Trade-off**: Balance between initial exploration and training time

## Training Loop Optimizations

### 9. Epsilon Decay
- **Status**: ✅ Implemented with multiple strategies
- **Constants**: `EPSILON_DECAY_TYPE`, `EPSILON_STEP_DECAY_EPISODES`, `EPSILON_STEP_DECAY_VALUES`
- **Decay types available**:
  - `exponential`: Standard exponential decay (default)
  - `linear`: Linear decay to minimum
  - `step`: Discrete steps at specified episodes
  - `cosine`: Cosine annealing with warm restarts
- **Effect of faster decay**: Quicker transition to exploitation, faster convergence, but may miss important explorations
- **Effect of slower decay**: More thorough exploration, better final policy, but slower convergence
- **Trade-off**: Balance between exploration and exploitation
- **Usage**: Configure via `epsilon_decay_type` parameter in WhistTrainer

### 10. Target Network Updates
- **Constant**: `UPDATE_TARGET_EVERY`
- **Effect of more frequent updates**: Faster adaptation to policy changes, but less stable Q-value estimates
- **Effect of less frequent updates**: More stable Q-values, but slower adaptation
- **Alternative**: Try soft updates with small τ parameter at each step
- **Trade-off**: Balance between stability and adaptability

### 11. Model Save Threshold
- **Status**: ✅ Implemented
- **Constant**: `MODEL_SAVE_REWARD_THRESHOLD`
- **Effect of higher (less negative) threshold**: Only better models saved, less disk usage, but may miss intermediate progress
- **Effect of lower (more negative) threshold**: More models saved including mediocre ones, better tracking of progress, but more disk usage
- **Related**: `MODEL_SAVE_CHECK_EVERY`, `MODEL_SAVE_MIN_GAMES`
- **Trade-off**: Balance between model quality and progress tracking
- **Impact**: Prevents saving poorly performing models

## Performance Optimizations

### 12. GPU Acceleration
- **Status**: ✅ Implemented
- **Current**: Automatic GPU detection with configurable memory limits
- **Configuration**: `USE_GPU`, `GPU_MEMORY_GROWTH`, `GPU_MEMORY_LIMIT_MB`
- **Impact**: 10-50x speedup for large models
- **Notes**: Mixed precision training available but experimental

### 13. Parallel Training
- **Status**: ❌ Not Implemented
- **Suggestion**: Use multiple environments in parallel (A3C/IMPALA style)
- **Implementation**: Create multiple game instances and collect experiences simultaneously
- **Impact**: Faster data collection, reduced training time
- **Complexity**: Requires significant refactoring

### 14. Prioritized Experience Replay
- **Status**: ❌ Not Implemented
- **Suggestion**: Sample experiences based on TD-error magnitude
- **Impact**: Learn more from important/surprising experiences
- **Implementation**: Maintain priority queue based on |TD-error|
- **Notes**: Adds complexity but can significantly speed up learning

## Simple Quick Wins

### 15. Reduce Logging
- **Status**: ✅ Already implemented
- **Notes**: Minimal output for production runs, detailed logs available

### 16. Early Stopping
- **Status**: ✅ Implemented
- **Parameters**: `early_stopping_patience` (number of episodes), `early_stopping_min_delta` (minimum improvement)
- **Effect of higher patience**: More tolerant of plateaus, longer training, better chance of finding improvements
- **Effect of lower patience**: Stops sooner, saves time, but may stop before finding improvements
- **Implementation**: Monitors rolling average reward and stops when no improvement for N episodes
- **Trade-off**: Balance between training time and finding optimal policy
- **Notes**: Only active after exploration phase, checks every MODEL_SAVE_CHECK_EVERY episodes, set patience to 0 to disable

### 17. Experience Diversity Tracking
- **Status**: ❌ Not Implemented
- **Suggestion**: Track state diversity in replay buffer to avoid repetitive experiences
- **Implementation**: Use state hashing or clustering to measure diversity
- **Impact**: Better exploration, more robust policies
- **Complexity**: Medium

## Advanced Optimizations

### 18. Double DQN
- **Status**: ✅ Implemented  
- **Parameter**: `use_double_dqn` in agent initialization
- **Implementation**: Decouples action selection (online network) from value estimation (target network)
- **Effect**: Reduces overestimation bias in Q-values, leading to more accurate value estimates
- **Impact**: More stable training, better final performance, minimal computational overhead
- **Trade-off**: Slightly more complex but proven benefits with negligible cost
- **Notes**: Enabled by default in both DQNAgent and EmbeddedDQNAgent

### 19. Dueling DQN
- **Status**: ✅ Implemented
- **Class**: `DuelingDQNAgent` in `advanced_dqn.py`
- **Architecture**: Separate value (V) and advantage (A) streams
- **Formula**: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
- **Effect**: Better learning in states where action choice matters less
- **Usage**: Set `use_dueling_dqn=True` in WhistTrainer
- **Notes**: Includes learning rate scheduling and n-step returns by default

### 20. Multi-Step Returns
- **Status**: ✅ Implemented
- **Class**: `NStepReplayBuffer` in `advanced_dqn.py`
- **Constant**: `N_STEP_RETURNS` (default: 3)
- **Effect**: Faster credit assignment, better bootstrapping
- **Implementation**: Accumulates rewards over N steps before computing TD target
- **Trade-off**: Higher N = faster propagation but higher variance
- **Notes**: Automatically used with DuelingDQNAgent, stores longer trajectories

### 21. Curriculum Learning
- **Status**: ❌ Not Implemented
- **Suggestion**: Gradually increase opponent difficulty or game complexity
- **Impact**: More stable learning, better final performance
- **Implementation**: Start with simpler opponents (more random play), gradually reduce randomness
- **Notes**: Could vary EW_RANDOM_PLAY_PROBABILITY during training

### 22. Self-Play Training
- **Status**: ❌ Not Implemented
- **Suggestion**: Train agents against previous versions of themselves
- **Impact**: Continuous improvement, discover novel strategies
- **Implementation**: Maintain pool of past model checkpoints, sample opponents from pool
- **Notes**: Particularly effective for competitive games

## Hyperparameter Tuning

### 23. Systematic Grid/Random Search
- **Status**: ❌ Not Implemented
- **Suggestion**: Systematically explore hyperparameter space
- **Parameters to tune**: learning rate, gamma, epsilon decay, network sizes, dropout rates
- **Tools**: Optuna, Ray Tune, or simple grid search
- **Impact**: Find optimal configuration for this specific task
- **Notes**: Time-consuming but can yield significant improvements

### 24. Adaptive Reward Scaling
- **Status**: ❌ Not Implemented
- **Suggestion**: Dynamically scale rewards based on training progress
- **Impact**: Maintain consistent learning signal throughout training
- **Implementation**: Normalize rewards by running statistics
- **Notes**: Can help with varying reward magnitudes

## Implementation Priority

**High Priority** (Quick wins):
- ✅ Batch size tuning (done)
- ✅ Trump play penalties (done)  
- ✅ GPU acceleration (done)
- ✅ Fix MODEL_SAVE_REWARD_THRESHOLD constant (done)
- ✅ Implement early stopping (done)
- ✅ Try Double DQN (done)

**Medium Priority** (Moderate effort, good impact):
- ✅ Increase replay memory size (done - increased from 100 to 50,000)
- ✅ Implement Dueling DQN (done - separate value and advantage streams)
- ✅ Add learning rate scheduling (done - exponential decay with configurable parameters)
- ✅ Tune epsilon decay schedule (done - multiple strategies: exponential, linear, step, cosine)
- ✅ Implement multi-step returns (done - N-step TD with configurable N)

**Low Priority** (High effort or experimental):
- Prioritized Experience Replay
- Parallel training (A3C/IMPALA)
- Curriculum learning
- Self-play training
- Systematic hyperparameter search

## Notes

- Start with one optimization at a time to measure impact
- Keep track of hyperparameters and results
- Use TensorBoard for monitoring training progress
- Consider trade-offs between training speed and sample efficiency
- Some optimizations may interact; test combinations carefully
