import numpy as np
import tensorflow as tf
from collections import deque
import random
from typing import List, Optional, Any, Tuple
from base_classes import BaseAgent
from constants import (
    DEFAULT_GAMMA,
    REPLAY_MEMORY_SIZE,
    MIN_REPLAY_MEMORY_SIZE,
    MINIBATCH_SIZE,
    UPDATE_TARGET_EVERY,
    MODEL_NAME,
    MIN_REWARD,
    MEMORY_FRACTION,
    ARRAY_LENGTH,
    GAME_INPUT_SIZE,
    PLAYER_INPUT_SIZE,
    TRACKING_INPUT_SIZE,
    SCORE_INPUT_SIZE,
    HIDDEN_LAYER_1_SIZE,
    HIDDEN_LAYER_2_SIZE,
    HIDDEN_LAYER_3_SIZE,
    DROPOUT_RATE
)

class DQNAgent(BaseAgent):
    def __init__(self, input_size: int, gamma, agent_id: int = 0):
        super().__init__(agent_id)
        self.input_shape = input_size
        self.model = self.create_model()
        self.gamma = gamma

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=100000)

        self.target_update_counter = 0

    def create_model(self):
        game_input = tf.keras.layers.Input(shape=(GAME_INPUT_SIZE,))
        player_input = tf.keras.layers.Input(shape=(PLAYER_INPUT_SIZE,))
        tracking_input = tf.keras.layers.Input(shape=(TRACKING_INPUT_SIZE,))
        score_input = tf.keras.layers.Input(shape=(SCORE_INPUT_SIZE,))

        game_features = tf.keras.layers.Dense(GAME_INPUT_SIZE, activation='relu')(game_input)
        player_features = tf.keras.layers.Dense(PLAYER_INPUT_SIZE, activation='relu')(player_input)
        tracking_features = tf.keras.layers.Dense(TRACKING_INPUT_SIZE, activation='relu')(tracking_input)
        score_features = tf.keras.layers.Dense(ARRAY_LENGTH, activation='relu')(score_input)

        combined = tf.keras.layers.Concatenate()([game_features, player_features, tracking_features, score_features])

        hidden1 = tf.keras.layers.Dense(HIDDEN_LAYER_1_SIZE, activation='relu')(combined)
        dropout1 = tf.keras.layers.Dropout(DROPOUT_RATE)(hidden1)
        hidden2 = tf.keras.layers.Dense(HIDDEN_LAYER_2_SIZE, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(DROPOUT_RATE)(hidden2)
        hidden3 = tf.keras.layers.Dense(HIDDEN_LAYER_3_SIZE, activation='relu')(dropout2)
        dropout3 = tf.keras.layers.Dropout(DROPOUT_RATE)(hidden3)

        # Output layer for Q-values
        output = tf.keras.layers.Dense(ARRAY_LENGTH, activation='linear')(dropout3)  # 13 possible card actions

        # Create model with multiple inputs
        model = tf.keras.Model(
            inputs=[game_input, player_input, tracking_input, score_input],
            outputs=output
        )

        model.compile(optimizer='adam', loss='mse', jit_compile=False)
        return model

    def update_replay_memory(self, transition):
        state, action, reward, next_state, done = transition
        # print(transition)
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        # Your existing training code, but modified to handle multiple inputs
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Process all states in batch
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

        # Get current Q values (using direct call instead of predict to avoid retracing)
        current_qs_list = self.model(
            [current_game_data, current_player_data, current_tracking_data, current_score_data],
            verbose=0,
            batch_size=MINIBATCH_SIZE
        )

        # Get future Q values
        future_qs_list = self.target_model(
            [new_game_data, new_player_data, new_tracking_data, new_score_data],
            verbose=0,
            batch_size=MINIBATCH_SIZE
        )

        X_game = []
        X_player = []
        X_tracking = []
        X_score = []
        y = []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = np.array(current_qs_list[index])
            current_qs[action] = new_q

            # And append to training data
            X_game.append(current_game_data[index])
            X_player.append(current_player_data[index])
            X_tracking.append(current_tracking_data[index])
            X_score.append(current_score_data[index])
            y.append(current_qs)

        # Fit on all samples as one batch
        self.model.fit(
            [np.array(X_game), np.array(X_player), np.array(X_tracking), np.array(X_score)],
            np.array(y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False if terminal_state else None
        )

        # Update target network if needed
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def _flat_the_state(self, state):
        if state is None:
            raise ValueError("Received None as state in _flat_the_state")
        flat_state = np.array([item for sublist in state for item in (sublist if isinstance(sublist, list) else [sublist])])
        return flat_state

    def get_qs(self, state):

        # Preprocess state into the correct input format
        game_data = np.concatenate([state[0], state[1]])  # cards_array + round_array
        player_data = np.concatenate([state[2], state[3]])  # hand_array + player_array
        tracking_data = np.concatenate([state[4], state[5], state[6], state[7]])  # all player card tracking
        score_data = np.array(state[8])  # score_array

        # Add batch dimension
        game_data = np.expand_dims(game_data, axis=0)
        player_data = np.expand_dims(player_data, axis=0)
        tracking_data = np.expand_dims(tracking_data, axis=0)
        score_data = np.expand_dims(score_data, axis=0)

        # Predict using all inputs with batch_size
        return self.model.predict(
            [game_data, player_data, tracking_data, score_data],
            verbose=0,
            batch_size=1
        )

    def predict_action(self, state):
        state_input = self._flat_the_state(state)  # Ensure correct shape
        q_values = self.model.predict(state_input, verbose=0)  # Get Q-values
        return np.argmax(q_values)  # Choose best action
    
    def choose_action(self, state: Any, valid_actions: List[int], epsilon: Optional[float] = None) -> int:
        """
        Choose an action using epsilon-greedy strategy.
        
        Args:
            state: The current game state
            valid_actions: List of valid action indices
            epsilon: Exploration rate (if None, always exploit)
            
        Returns:
            The chosen action index
        """
        if not valid_actions:
            return 0
        
        if epsilon is not None and np.random.random() < epsilon:
            # Explore: choose random valid action
            return np.random.choice(valid_actions)
        else:
            # Exploit: choose best valid action based on Q-values
            qs = self.get_qs(state)[0]
            valid_qs = [(action, qs[action]) for action in valid_actions if action < len(qs)]
            if valid_qs:
                return max(valid_qs, key=lambda x: x[1])[0]
            return valid_actions[0]
    
    def update(self, transition: Tuple) -> None:
        """
        Update the agent with a new transition (for BaseAgent compatibility).
        
        Args:
            transition: Tuple of (state, action, reward, next_state, done)
        """
        self.update_replay_memory(transition)

    @staticmethod
    def save_agent(agent, filename):
        agent.model.save_weights(filename)
        print(f"Agent weights saved to {filename}")

    @staticmethod
    def save_full_agent(agent, filename):
        agent.model.save(filename)
        print(f"Full agent model saved to {filename}")
