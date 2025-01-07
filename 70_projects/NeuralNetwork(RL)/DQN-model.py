import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(state_size, action_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(state_size,)),
        Dense(64, activation='relu'),
        Dense(action_size, activation='linear')  # Output Q-values for each action
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = build_model(state_size, action_size)
        self.target_model = build_model(state_size, action_size)
        self.memory = []  # Replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        q_values = self.model.predict(state[np.newaxis, :])  # Exploit
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.target_model.predict(next_state[np.newaxis, :]))
            target_f = self.model.predict(state[np.newaxis, :])
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = WhistEnv()
agent = DQNAgent(state_size=10, action_size=len(env.action_space))

episodes = 1000
for e in range(episodes):
    state = env.reset()
    total_reward = 0

    for time in range(100):  # Limit each game to 100 steps
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    agent.replay()
    agent.update_target_model()
    agent.decay_epsilon()
    print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")
