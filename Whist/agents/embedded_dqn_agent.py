"""
Embedded DQN Agent for Whist

This agent uses card embeddings instead of one-hot encoding for a more
flexible and expressive representation that works with any number of cards.
"""

import numpy as np
import tensorflow as tf
from collections import deque
import random
from typing import List, Optional, Any, Tuple
from Whist.utils.base_classes import BaseAgent
from Whist.embedding.card_embedding import CardEmbedding
from Whist.utils.constants import (
    DEFAULT_GAMMA,
    REPLAY_MEMORY_SIZE,
    MIN_REPLAY_MEMORY_SIZE,
    MINIBATCH_SIZE,
    UPDATE_TARGET_EVERY,
    CARDS_PER_PLAYER
)


class EmbeddedDQNAgent(BaseAgent):
    """
    DQN Agent using card embeddings instead of one-hot encoding.
    
    This agent is flexible to different numbers of cards and uses learned
    embeddings to represent cards, hands, and game state.
    """
    
    def __init__(self, embedding_dim: int = 8, gamma: float = 0.99, agent_id: int = 0):
        """
        Initialize the Embedded DQN Agent.
        
        Args:
            embedding_dim: Dimension of card embedding vectors (default: 8)
            gamma: Discount factor for future rewards
            agent_id: Unique identifier for this agent
        """
        super().__init__(agent_id)
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        
        # Create card embedding layer (shared between model and target)
        self.card_embedding = CardEmbedding(embedding_dim=embedding_dim)
        
        # Create models
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=100000)
        self.target_update_counter = 0
    
    def create_model(self):
        """
        Create the neural network model using embeddings.
        
        State components:
        - Hand embedding (aggregated): embedding_dim
        - Round embedding (aggregated): embedding_dim
        - Played cards embedding (aggregated): embedding_dim
        - Player encoding (one-hot): 4
        - Player tracking embeddings (4 players): embedding_dim * 4
        - Scores: 4
        
        Total: embedding_dim * 7 + 8
        """
        # Input layers for different state components
        hand_input = tf.keras.layers.Input(shape=(self.embedding_dim,), name='hand')
        round_input = tf.keras.layers.Input(shape=(self.embedding_dim,), name='round')
        played_input = tf.keras.layers.Input(shape=(self.embedding_dim,), name='played')
        player_input = tf.keras.layers.Input(shape=(4,), name='player_id')
        tracking_input = tf.keras.layers.Input(shape=(self.embedding_dim * 4,), name='tracking')
        score_input = tf.keras.layers.Input(shape=(4,), name='scores')
        
        # Process each input
        hand_features = tf.keras.layers.Dense(32, activation='relu')(hand_input)
        round_features = tf.keras.layers.Dense(32, activation='relu')(round_input)
        played_features = tf.keras.layers.Dense(32, activation='relu')(played_input)
        player_features = tf.keras.layers.Dense(16, activation='relu')(player_input)
        tracking_features = tf.keras.layers.Dense(64, activation='relu')(tracking_input)
        score_features = tf.keras.layers.Dense(16, activation='relu')(score_input)
        
        # Concatenate all features
        combined = tf.keras.layers.Concatenate()([
            hand_features, round_features, played_features,
            player_features, tracking_features, score_features
        ])
        
        # Hidden layers
        hidden1 = tf.keras.layers.Dense(128, activation='relu')(combined)
        dropout1 = tf.keras.layers.Dropout(0.3)(hidden1)
        hidden2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.3)(hidden2)
        hidden3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)
        
        # Output layer - Q-values for each possible card in hand
        # Note: CARDS_PER_PLAYER is max hand size, actual valid actions determined at runtime
        output = tf.keras.layers.Dense(CARDS_PER_PLAYER, activation='linear')(hidden3)
        
        # Create model
        model = tf.keras.Model(
            inputs=[hand_input, round_input, played_input, player_input, tracking_input, score_input],
            outputs=output,
            name='embedded_dqn'
        )
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def update_replay_memory(self, transition):
        """Add transition to replay memory."""
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
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        # Prepare batch data
        current_states = {
            'hand': [], 'round': [], 'played': [],
            'player_id': [], 'tracking': [], 'scores': []
        }
        next_states = {
            'hand': [], 'round': [], 'played': [],
            'player_id': [], 'tracking': [], 'scores': []
        }
        
        for state, action, reward, next_state, done in minibatch:
            # Current state
            for key, value in zip(['hand', 'round', 'played', 'player_id', 'tracking', 'scores'], state):
                current_states[key].append(value)
            
            # Next state
            for key, value in zip(['hand', 'round', 'played', 'player_id', 'tracking', 'scores'], next_state):
                next_states[key].append(value)
        
        # Convert to numpy arrays
        for key in current_states:
            current_states[key] = np.array(current_states[key])
            next_states[key] = np.array(next_states[key])
        
        # Get current Q values
        current_qs_list = self.model.predict(
            [current_states['hand'], current_states['round'], current_states['played'],
             current_states['player_id'], current_states['tracking'], current_states['scores']],
            verbose=0
        )
        
        # Get future Q values
        future_qs_list = self.target_model.predict(
            [next_states['hand'], next_states['round'], next_states['played'],
             next_states['player_id'], next_states['tracking'], next_states['scores']],
            verbose=0
        )
        
        X = {key: [] for key in current_states.keys()}
        y = []
        
        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward
            
            current_qs = np.array(current_qs_list[index])
            current_qs[action] = new_q
            
            # Add to training data
            for i, key in enumerate(['hand', 'round', 'played', 'player_id', 'tracking', 'scores']):
                X[key].append(current_states[key][index])
            y.append(current_qs)
        
        # Convert to numpy
        for key in X:
            X[key] = np.array(X[key])
        y = np.array(y)
        
        # Fit model
        self.model.fit(
            [X['hand'], X['round'], X['played'], X['player_id'], X['tracking'], X['scores']],
            y,
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False
        )
        
        # Update target network
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        """
        Get Q-values for a given state.
        
        Args:
            state: List of [hand_emb, round_emb, played_emb, player_encoding, tracking_embs, scores]
        
        Returns:
            Q-values for each action
        """
        # Unpack state
        hand_emb, round_emb, played_emb, player_encoding, tracking_embs, scores = state
        
        # Add batch dimension
        hand_emb = np.expand_dims(hand_emb, axis=0)
        round_emb = np.expand_dims(round_emb, axis=0)
        played_emb = np.expand_dims(played_emb, axis=0)
        player_encoding = np.expand_dims(player_encoding, axis=0)
        tracking_embs = np.expand_dims(tracking_embs, axis=0)
        scores = np.expand_dims(scores, axis=0)
        
        # Predict
        return self.model.predict(
            [hand_emb, round_emb, played_emb, player_encoding, tracking_embs, scores],
            verbose=0
        )
    
    def choose_action(self, state: Any, valid_actions: List[int], epsilon: Optional[float] = None, return_info: bool = False):
        """
        Choose an action using epsilon-greedy strategy.
        
        Args:
            state: The current embedded state
            valid_actions: List of valid action indices
            epsilon: Exploration rate
            return_info: If True, return (action, is_exploration, q_value) tuple
        
        Returns:
            The chosen action index, or tuple (action, is_exploration, q_value) if return_info=True
        """
        if not valid_actions:
            if return_info:
                return 0, False, None
            return 0
        
        if epsilon is not None and np.random.random() < epsilon:
            # Explore
            action = np.random.choice(valid_actions)
            if return_info:
                return action, True, None
            return action
        else:
            # Exploit
            qs = self.get_qs(state)[0]
            valid_qs = [(action, qs[action]) for action in valid_actions if action < len(qs)]
            if valid_qs:
                best_action, best_q = max(valid_qs, key=lambda x: x[1])
                if return_info:
                    return best_action, False, float(best_q)
                return best_action
            if return_info:
                return valid_actions[0], False, None
            return valid_actions[0]
    
    def update(self, transition: Tuple) -> None:
        """Update replay memory with transition."""
        self.update_replay_memory(transition)
    
    def save_agent(self, filename):
        """Save agent weights."""
        self.model.save_weights(filename)
        print(f"Agent weights saved to {filename}")

    def save_full_agent(self, filename):
        """Save full agent model."""
        self.model.save(filename)
        print(f"Full agent model saved to {filename}")
