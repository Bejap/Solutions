"""
Advanced DQN Architectures

This module provides advanced DQN architectures including:
- Dueling DQN: Separates state value and action advantage estimation
- Multi-step returns: N-step TD learning for faster credit assignment
- Learning rate scheduling: Configurable LR decay strategies
"""

import numpy as np
import tensorflow as tf
from collections import deque
import random
from typing import List, Optional, Any, Tuple
from Whist.utils.base_classes import BaseAgent
from Whist.utils.constants import (
    DEFAULT_GAMMA,
    REPLAY_MEMORY_SIZE,
    MIN_REPLAY_MEMORY_SIZE,
    MINIBATCH_SIZE,
    UPDATE_TARGET_EVERY,
    ARRAY_LENGTH,
    ACTION_SIZE,
    GAME_INPUT_SIZE,
    PLAYER_INPUT_SIZE,
    TRACKING_INPUT_SIZE,
    SCORE_INPUT_SIZE,
    HIDDEN_LAYER_1_SIZE,
    HIDDEN_LAYER_2_SIZE,
    HIDDEN_LAYER_3_SIZE,
    DROPOUT_RATE,
    USE_GPU,
    GPU_MEMORY_GROWTH,
    GPU_MEMORY_LIMIT_MB,
    USE_MIXED_PRECISION,
    N_STEP_RETURNS,
    INITIAL_LEARNING_RATE,
    LR_DECAY_STEPS,
    LR_DECAY_RATE,
    MIN_LEARNING_RATE
)
from Whist.utils.device_config import configure_device, enable_mixed_precision


class NStepReplayBuffer:
    """
    N-step replay buffer for multi-step returns.
    
    Stores transitions and computes n-step returns for improved
    credit assignment and faster learning.
    """
    
    def __init__(self, maxlen: int = REPLAY_MEMORY_SIZE, n_steps: int = N_STEP_RETURNS, gamma: float = 0.99):
        """
        Initialize n-step replay buffer.
        
        Args:
            maxlen: Maximum buffer size
            n_steps: Number of steps for returns calculation
            gamma: Discount factor
        """
        self.buffer = deque(maxlen=maxlen)
        self.n_step_buffer = deque(maxlen=n_steps)
        self.n_steps = n_steps
        self.gamma = gamma
    
    def append(self, transition: Tuple):
        """
        Add transition to buffer, computing n-step returns when possible.
        
        Args:
            transition: Tuple of (state, action, reward, next_state, done)
        """
        self.n_step_buffer.append(transition)
        
        # When we have enough transitions, compute n-step return
        if len(self.n_step_buffer) == self.n_steps:
            n_step_return = self._compute_n_step_return()
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]
            
            self.buffer.append((first_state, first_action, n_step_return, last_state, last_done))
        
        # Handle terminal states - flush remaining transitions
        if transition[4]:  # done flag
            self._flush_n_step_buffer()
    
    def _compute_n_step_return(self) -> float:
        """Compute the n-step return from buffered transitions."""
        n_step_return = 0.0
        for i, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * reward
        return n_step_return
    
    def _flush_n_step_buffer(self):
        """Flush remaining transitions in n-step buffer at episode end."""
        while len(self.n_step_buffer) > 0:
            # Compute return for remaining transitions
            n_step_return = 0.0
            for i, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * reward
            
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]
            
            self.buffer.append((first_state, first_action, n_step_return, last_state, last_done))
            self.n_step_buffer.popleft()
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent(BaseAgent):
    """
    Dueling DQN Agent with separate value and advantage streams.
    
    Architecture separates Q-value estimation into:
    - V(s): State value function
    - A(s, a): Action advantage function
    
    Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
    
    This helps the agent learn which states are valuable independent
    of which action is taken.
    """
    
    def __init__(self, input_size: int, gamma: float = DEFAULT_GAMMA, agent_id: int = 0,
                 use_double_dqn: bool = True, use_n_step: bool = True,
                 use_lr_scheduling: bool = True):
        """
        Initialize the Dueling DQN Agent.
        
        Args:
            input_size: Size of flattened input
            gamma: Discount factor for future rewards
            agent_id: Unique identifier for this agent
            use_double_dqn: Use Double DQN algorithm (default: True)
            use_n_step: Use n-step returns (default: True)
            use_lr_scheduling: Use learning rate scheduling (default: True)
        """
        super().__init__(agent_id)
        self.input_shape = input_size
        self.gamma = gamma
        self.use_double_dqn = use_double_dqn
        self.use_n_step = use_n_step
        self.use_lr_scheduling = use_lr_scheduling
        self.training_step = 0
        
        # Configure GPU/NPU if requested (only once per process)
        if USE_GPU and not hasattr(DuelingDQNAgent, '_device_configured'):
            self.device = configure_device(
                prefer_gpu=USE_GPU,
                memory_growth=GPU_MEMORY_GROWTH,
                memory_limit_mb=GPU_MEMORY_LIMIT_MB
            )
            if USE_MIXED_PRECISION:
                enable_mixed_precision()
            DuelingDQNAgent._device_configured = True
        
        # Learning rate scheduler
        if use_lr_scheduling:
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=INITIAL_LEARNING_RATE,
                decay_steps=LR_DECAY_STEPS,
                decay_rate=LR_DECAY_RATE,
                staircase=True
            )
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        else:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
        
        # Create models
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        # Replay memory - use n-step buffer if enabled
        if use_n_step:
            self.replay_memory = NStepReplayBuffer(
                maxlen=REPLAY_MEMORY_SIZE,
                n_steps=N_STEP_RETURNS,
                gamma=gamma
            )
        else:
            self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.target_update_counter = 0
    
    def create_model(self):
        """
        Create Dueling DQN model with separate value and advantage streams.
        """
        # Input layers
        game_input = tf.keras.layers.Input(shape=(GAME_INPUT_SIZE,), name='game')
        player_input = tf.keras.layers.Input(shape=(PLAYER_INPUT_SIZE,), name='player')
        tracking_input = tf.keras.layers.Input(shape=(TRACKING_INPUT_SIZE,), name='tracking')
        score_input = tf.keras.layers.Input(shape=(SCORE_INPUT_SIZE,), name='score')
        
        # Process each input
        game_features = tf.keras.layers.Dense(GAME_INPUT_SIZE, activation='relu')(game_input)
        player_features = tf.keras.layers.Dense(PLAYER_INPUT_SIZE, activation='relu')(player_input)
        tracking_features = tf.keras.layers.Dense(TRACKING_INPUT_SIZE, activation='relu')(tracking_input)
        score_features = tf.keras.layers.Dense(ARRAY_LENGTH, activation='relu')(score_input)
        
        # Combine features
        combined = tf.keras.layers.Concatenate()([
            game_features, player_features, tracking_features, score_features
        ])
        
        # Shared hidden layers
        hidden1 = tf.keras.layers.Dense(HIDDEN_LAYER_1_SIZE, activation='relu')(combined)
        dropout1 = tf.keras.layers.Dropout(DROPOUT_RATE)(hidden1)
        hidden2 = tf.keras.layers.Dense(HIDDEN_LAYER_2_SIZE, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(DROPOUT_RATE)(hidden2)
        
        # === DUELING ARCHITECTURE ===
        
        # Value stream (V(s))
        value_hidden = tf.keras.layers.Dense(HIDDEN_LAYER_3_SIZE, activation='relu')(dropout2)
        value_output = tf.keras.layers.Dense(1, activation='linear', name='value')(value_hidden)
        
        # Advantage stream (A(s, a))
        advantage_hidden = tf.keras.layers.Dense(HIDDEN_LAYER_3_SIZE, activation='relu')(dropout2)
        advantage_output = tf.keras.layers.Dense(ACTION_SIZE, activation='linear', name='advantage')(advantage_hidden)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        # This aggregation helps identifiability of V and A
        def aggregate(inputs):
            value, advantage = inputs
            # Subtract mean advantage to improve stability
            return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        q_values = tf.keras.layers.Lambda(aggregate, name='q_values')([value_output, advantage_output])
        
        # Create model
        model = tf.keras.Model(
            inputs=[game_input, player_input, tracking_input, score_input],
            outputs=q_values,
            name='dueling_dqn'
        )
        
        model.compile(optimizer=self.optimizer, loss='mse', jit_compile=False)
        return model
    
    def update_replay_memory(self, transition):
        """Add transition to replay memory."""
        if self.use_n_step:
            self.replay_memory.append(transition)
        else:
            self.replay_memory.append(transition)
    
    def train(self, terminal_state: bool, step: int):
        """
        Train the agent on a minibatch from replay memory.
        
        Args:
            terminal_state: Whether this is a terminal state
            step: Current step count
        """
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = self.replay_memory.sample(MINIBATCH_SIZE) if self.use_n_step else random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        # Process batch
        current_game_data = []
        current_player_data = []
        current_tracking_data = []
        current_score_data = []
        
        new_game_data = []
        new_player_data = []
        new_tracking_data = []
        new_score_data = []
        
        for state, action, reward, next_state, done in minibatch:
            # Current state
            current_game_data.append(np.concatenate([state[0], state[1]]))
            current_player_data.append(np.concatenate([state[2], state[3]]))
            current_tracking_data.append(np.concatenate([state[4], state[5], state[6], state[7]]))
            current_score_data.append(np.array(state[8]))
            
            # Next state
            new_game_data.append(np.concatenate([next_state[0], next_state[1]]))
            new_player_data.append(np.concatenate([next_state[2], next_state[3]]))
            new_tracking_data.append(np.concatenate([next_state[4], next_state[5], next_state[6], next_state[7]]))
            new_score_data.append(np.array(next_state[8]))
        
        # Convert to numpy arrays
        current_game_data = np.array(current_game_data)
        current_player_data = np.array(current_player_data)
        current_tracking_data = np.array(current_tracking_data)
        current_score_data = np.array(current_score_data)
        
        new_game_data = np.array(new_game_data)
        new_player_data = np.array(new_player_data)
        new_tracking_data = np.array(new_tracking_data)
        new_score_data = np.array(new_score_data)
        
        # Get current Q values
        current_qs_list = self.model(
            [current_game_data, current_player_data, current_tracking_data, current_score_data],
            training=False
        )
        
        # Get future Q values (Double DQN if enabled)
        if self.use_double_dqn:
            future_qs_online = self.model(
                [new_game_data, new_player_data, new_tracking_data, new_score_data],
                training=False
            )
            future_qs_target = self.target_model(
                [new_game_data, new_player_data, new_tracking_data, new_score_data],
                training=False
            )
        else:
            future_qs_list = self.target_model(
                [new_game_data, new_player_data, new_tracking_data, new_score_data],
                training=False
            )
        
        # Prepare training data
        X_game = []
        X_player = []
        X_tracking = []
        X_score = []
        y = []
        
        # Discount factor for n-step returns (already applied in buffer if using n-step)
        effective_gamma = self.gamma ** N_STEP_RETURNS if self.use_n_step else self.gamma
        
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                if self.use_double_dqn:
                    best_action = np.argmax(future_qs_online[index])
                    max_future_q = future_qs_target[index][best_action]
                else:
                    max_future_q = np.max(future_qs_list[index])
                new_q = reward + effective_gamma * max_future_q
            else:
                new_q = reward
            
            current_qs = np.array(current_qs_list[index])
            current_qs[action] = new_q
            
            X_game.append(current_game_data[index])
            X_player.append(current_player_data[index])
            X_tracking.append(current_tracking_data[index])
            X_score.append(current_score_data[index])
            y.append(current_qs)
        
        # Fit model
        self.model.fit(
            [np.array(X_game), np.array(X_player), np.array(X_tracking), np.array(X_score)],
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False
        )
        
        self.training_step += 1
        
        # Update target network
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        """Get Q-values for a given state."""
        game_data = np.concatenate([state[0], state[1]])
        player_data = np.concatenate([state[2], state[3]])
        tracking_data = np.concatenate([state[4], state[5], state[6], state[7]])
        score_data = np.array(state[8])
        
        game_data = np.expand_dims(game_data, axis=0)
        player_data = np.expand_dims(player_data, axis=0)
        tracking_data = np.expand_dims(tracking_data, axis=0)
        score_data = np.expand_dims(score_data, axis=0)
        
        return self.model.predict(
            [game_data, player_data, tracking_data, score_data],
            verbose=0
        )
    
    def choose_action(self, state: Any, valid_actions: List[int], epsilon: Optional[float] = None) -> int:
        """Choose an action using epsilon-greedy strategy."""
        if not valid_actions:
            return 0
        
        if epsilon is not None and np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        else:
            qs = self.get_qs(state)[0]
            valid_qs = [(action, qs[action]) for action in valid_actions if action < len(qs)]
            if valid_qs:
                return max(valid_qs, key=lambda x: x[1])[0]
            return valid_actions[0]
    
    def update(self, transition: Tuple) -> None:
        """Update replay memory with transition."""
        self.update_replay_memory(transition)
    
    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        if self.use_lr_scheduling:
            return self.lr_schedule(self.training_step).numpy()
        return INITIAL_LEARNING_RATE
    
    def save_agent(self, filename):
        """Save agent weights."""
        self.model.save_weights(filename)
        print(f"Agent weights saved to {filename}")

    def save_full_agent(self, filename):
        """Save full agent model."""
        self.model.save(filename)
        print(f"Full agent model saved to {filename}")


class EpsilonScheduler:
    """
    Flexible epsilon scheduling for exploration vs exploitation trade-off.
    
    Supports multiple decay strategies:
    - exponential: Standard exponential decay
    - linear: Linear decay to minimum
    - step: Discrete steps at specified episodes
    - cosine: Cosine annealing with warm restarts
    """
    
    def __init__(self, initial_epsilon: float = 1.0, min_epsilon: float = 0.01,
                 decay_type: str = 'exponential', decay_rate: float = 0.995,
                 total_episodes: int = 1000, step_episodes: List[int] = None,
                 step_values: List[float] = None, warm_restart_period: int = 200):
        """
        Initialize epsilon scheduler.
        
        Args:
            initial_epsilon: Starting epsilon value
            min_epsilon: Minimum epsilon floor
            decay_type: Type of decay ('exponential', 'linear', 'step', 'cosine')
            decay_rate: Decay rate for exponential decay
            total_episodes: Total expected episodes (for linear/cosine)
            step_episodes: Episodes at which to change epsilon (for step decay)
            step_values: Epsilon values at each step
            warm_restart_period: Period for cosine annealing warm restarts
        """
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.total_episodes = total_episodes
        self.step_episodes = step_episodes or [300, 600, 900]
        self.step_values = step_values or [0.5, 0.2, 0.05]
        self.warm_restart_period = warm_restart_period
        
        self.current_epsilon = initial_epsilon
        self.episode = 0
    
    def get_epsilon(self, episode: int = None) -> float:
        """
        Get epsilon value for the given episode.
        
        Args:
            episode: Current episode number (uses internal counter if None)
            
        Returns:
            Epsilon value for the episode
        """
        if episode is not None:
            self.episode = episode
        
        if self.decay_type == 'exponential':
            self.current_epsilon = max(
                self.min_epsilon,
                self.initial_epsilon * (self.decay_rate ** self.episode)
            )
        
        elif self.decay_type == 'linear':
            decay_per_episode = (self.initial_epsilon - self.min_epsilon) / self.total_episodes
            self.current_epsilon = max(
                self.min_epsilon,
                self.initial_epsilon - decay_per_episode * self.episode
            )
        
        elif self.decay_type == 'step':
            self.current_epsilon = self.initial_epsilon
            for ep, val in zip(self.step_episodes, self.step_values):
                if self.episode >= ep:
                    self.current_epsilon = val
            self.current_epsilon = max(self.min_epsilon, self.current_epsilon)
        
        elif self.decay_type == 'cosine':
            # Cosine annealing with warm restarts
            episode_in_period = self.episode % self.warm_restart_period
            cosine_factor = (1 + np.cos(np.pi * episode_in_period / self.warm_restart_period)) / 2
            self.current_epsilon = self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * cosine_factor
        
        return self.current_epsilon
    
    def step(self) -> float:
        """Advance to next episode and return new epsilon."""
        self.episode += 1
        return self.get_epsilon()
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_epsilon = self.initial_epsilon
        self.episode = 0
