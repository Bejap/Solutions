import keras
import numpy as np
import tensorflow as tf
from collections import deque
import random

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 500  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
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
        output_dim = 13  # Number of possible actions
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation="linear")
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def update_replay_memory(self, transition: list):
        if transition[4] is True:  # Ensure state is valid
            return
        # print(transition)
        self.replay_memory.append(transition)
    def train(self, terminal_state: list, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Sample a single transition
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)[0]

        # Unpack the single transition
        current_state, action, reward, new_current_state, done = minibatch

        # Flatten and prepare states
        flat_current_state = self._flat_the_state(current_state)
        flat_current_state = np.array(flat_current_state, dtype=np.float32).reshape(1, -1)

        flat_new_current_state = self._flat_the_state(new_current_state)
        flat_new_current_state = np.array(flat_new_current_state, dtype=np.float32).reshape(1, -1)

        # Predict Q-values
        current_qs_list = self.model.predict(flat_current_state)
        future_qs_list = self.target_model.predict(flat_new_current_state)

        # Calculate new Q-value
        if not done:
            max_future_q = np.max(future_qs_list[0])
            new_q = reward + GAMMA * max_future_q
        else:
            new_q = reward

        # Update Q-values
        current_qs = current_qs_list[0]
        current_qs[action] = new_q

        # Prepare training data
        x = flat_current_state
        y = current_qs.reshape(1, -1)

        # Train the model
        self.model.fit(x, y, epochs=1, verbose=0)

        # Update target model periodically
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
        # Flatten state to a single list
        flat_state = self._flat_the_state(state)
        # Debugging: Print flattened state to check if it looks right
        # print("Flattened state:", flattened_state)

        # Convert to NumPy array and predict
        return self.model.predict(np.array(flat_state).reshape(1, -1))[0]

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


# Load environment and agent
# env = whist.Whist(["1", "2", "3", "4"])  # Initialize the game environment
# my_agent = load_full_agent()
#
# Show model summary
# my_agent.model.summary()
#
# Run tests
# test_agent(my_agent, env)