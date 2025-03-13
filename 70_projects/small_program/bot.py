import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
import random


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)





def build_dqn_model(input_shape, num_actions):
    model = Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model