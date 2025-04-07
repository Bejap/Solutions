import numpy as np
import tensorflow as tf
from collections import deque
import random

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 100  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 300  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 6  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'smalle'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.35

class DQNAgent:
    def __init__(self, input_size: int):
        self.input_shape = input_size
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=100000)

        self.target_update_counter = 0

    def create_model(self):
        game_input = tf.keras.layers.Input(shape=(16,))
        player_input = tf.keras.layers.Input(shape=(12,))
        tracking_input = tf.keras.layers.Input(shape=(32,))
        score_input = tf.keras.layers.Input(shape=(4,))

        game_features = tf.keras.layers.Dense(16, activation='relu')(game_input)
        player_features = tf.keras.layers.Dense(16, activation='relu')(player_input)
        tracking_features = tf.keras.layers.Dense(32, activation='relu')(tracking_input)
        score_features = tf.keras.layers.Dense(8, activation='relu')(score_input)

        combined = tf.keras.layers.Concatenate()([game_features, player_features, tracking_features, score_features])

        hidden1 = tf.keras.layers.Dense(128, activation='relu')(combined)
        hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
        hidden3 = tf.keras.layers.Dense(32, activation='relu')(hidden2)

        # Output layer for Q-values
        output = tf.keras.layers.Dense(8, activation='linear')(hidden3)  # 13 possible card actions

        # Create model with multiple inputs
        model = tf.keras.Model(
            inputs=[game_input, player_input, tracking_input, score_input],
            outputs=output
        )

        model.compile(optimizer='adam', loss='mse')
        return model

    def update_replay_memory(self, transition):
        state, action, reward, next_state, done = transition
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

        # Get current Q values
        current_qs_list = self.model.predict(
            [current_game_data, current_player_data, current_tracking_data, current_score_data],
            verbose=0
        )

        # Get future Q values
        future_qs_list = self.target_model.predict(
            [new_game_data, new_player_data, new_tracking_data, new_score_data],
            verbose=0
        )

        X_game = []
        X_player = []
        X_tracking = []
        X_score = []
        y = []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index].copy()
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

        # Predict using all inputs
        return self.model.predict(
            [game_data, player_data, tracking_data, score_data],
            verbose=0
        )[0]

    def predict_action(self, state):
        state_input = self._flat_the_state(state)  # Ensure correct shape
        q_values = self.model.predict(state_input, verbose=0)  # Get Q-values
        return np.argmax(q_values)  # Choose best action

    def save_agent(agent, filename):
        agent.model.save_weights(filename)
        print(f"Agent weights saved to {filename}")

    def save_full_agent(agent, filename):
        agent.model.save(filename)
        print(f"Full agent model saved to {filename}")


def load_full_agent(filename="whist_dqn_agent"):
    loaded_model = tf.keras.models.load_model(filename)
    print("Full agent model loaded successfully")
    return DQNAgent(model=loaded_model)  # Wrap in agent class

def test_agent(agent, env, episodes=10):
    for episode in range(episodes):
        state = env.reset()  # Reset game for new episode
        done = False
        total_reward = 0

        while not done:
            action = agent.predict_action(state)  # Get action
            next_state, reward, done = env.step(action)  # Take action

            total_reward += reward
            state = next_state  # Move to next state

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


def qs_values_get(state, model):
    return model.get_qs(state)


# Load environment and agent
# env = whist.Whist(["1", "2", "3", "4"])  # Initialize the game environment
# my_agent = load_full_agent()
#
# Show model summary
# my_agent.model.summary()
#
# Run tests
# test_agent(my_agent, env)