import numpy as np
import tensorflow as tf
from collections import deque
import random

GAMMA = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'smalle'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.35

class DQNAgent:


    def __init__(self, input_size):
        self.input_dim = input_size
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=100000)

        self.target_update_counter = 0

    def create_model(self):
        output_dim = 12
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation="linear")
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):

        if len(self.replay_memory) < 100000:
            return

        minibatch = random.sample(self.replay_memory, k=1)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + GAMMA * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=minibatch, epochs=1, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > 100:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        # Flatten state to a single list
        flattened_state = [item for sublist in state for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Debugging: Print flattened state to check if it looks right
        print("Flattened state:", flattened_state)

        # Convert to NumPy array and predict
        return self.model.predict(np.array(flattened_state).reshape(1, -1))[0]

