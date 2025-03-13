import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
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


class CardGameEnv:
    def __init__(self):
        self.state = None
        self.done = False

    def reset(self):
        self.state = np.zeros(10)
        self.done = False
        return self.state


    def step(self, action):
        reward = 0
        self.done = True if some_condition else False
        return self.state, reward, self.done

    def get_possible_actions(self):
        return [0, 1, 2]


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.9

        self.model = build_dqn_model(state_dim, action_dim)
        self.target_model = build_dqn_model(state_dim, action_dim)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_buffer = ReplayBuffer(max_size=10000)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(state[np.newaxis])
        return np.argmax(q_values[0])

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, action, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        q_values_next = np.amax(self.target_model.predict(next_states), axis=1)
        targets = rewards + (1 -np.array(dones)) * self.gamma * q_values_next

        q_values = self.model.predict(states)

        for i in range(batch_size):
            q_values[i][action[i]] = targets[i]

        self.model.train_on_batch(states, q_values)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())




def build_dqn_model(state_dim, action_dim):
    model = Sequential([
        layers.Dense(128, activation='relu', input_shape=state_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_dim, activation='linear'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model


env = CardGameEnv()
agent = DQNAgent(state_dim=10, action_dim=3)

episodes = 500

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state)

        next_state, reward, done = env.step(action)

        agent.replay_buffer.add((state, action, reward, next_state, done))

        agent.train(batch_size=32)

        state = next_state
        total_reward += reward

        if done:
            break

    agent.update_target_network()

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")


# player1 = Agent([13, 10, 3])
# player2 = Agent([2, 6, 9])
# player3 = Agent([4, 7, 8])
# player4 = Agent([11, 12, 5])
