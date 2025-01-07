import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import random
from collections import deque
from tensorflow.keras.models import load_model

class KasinoEnv(gym.Env):
    def __init__(self):
        super(KasinoEnv, self).__init__()

        # State: Player's hand (4 cards) + table cards (4 cards)
        self.observation_space = spaces.Box(low=1, high=13, shape=(8,), dtype=np.int32)

        # Action: Play one of the 4 cards from the player's hand
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self):
        # Randomly deal 4 cards to the player and 4 to the table
        self.player_hand = np.random.randint(1, 14, size=4)
        self.table_cards = np.random.randint(1, 14, size=4)
        self.state = np.concatenate([self.player_hand, self.table_cards])
        return self.state

    def step(self, action):
        card_played = self.player_hand[action]
        reward = 0

        # Base capture
        matches = np.where(self.table_cards == card_played)[0]
        if len(matches) > 0:
            reward += len(matches) * 2  # More reward for multiple captures

        # Building combinations (same value cards)
        if len(matches) >= 2:
            reward += 5  # Bonus for multiple card capture

        # Penalize invalid moves more heavily
        if len(matches) == 0:
            reward -= 2

        # Update game state
        if len(matches) > 0:
            self.table_cards = np.delete(self.table_cards, matches)
        self.player_hand[action] = 0

        while len(self.table_cards) < 4:
            self.table_cards = np.append(self.table_cards, 0)

        self.state = np.concatenate([self.player_hand, self.table_cards])
        done = np.all(self.player_hand == 0)

        return self.state, reward, done, {}

    def render(self):
        print(f"Player Hand: {self.player_hand}")
        print(f"Table Cards: {self.table_cards}")




class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation="relu"),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = KasinoEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
episodes = 100
batch_size = 32

# for e in range(episodes):
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])
#     if e % 10 == 0:
#         agent.model.save(f'my_model_{e}.keras')
#         print(f"Model saved at episode {e}")
#
#     for time in range(500):
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_size])
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             print(f"Episode {e+1}/{episodes} - Score: {reward}, Epsilon: {agent.epsilon:.2}")
#             break
#
#     agent.replay(batch_size)


def human_vs_ai():
    env = KasinoEnv()
    state = env.reset()
    model = load_model('my_model_90.keras')
    done = False

    while not done:
        print("\nTable Cards:", env.table_cards)
        print("Your Hand:", env.player_hand)

        # Only show non-zero cards as valid moves
        valid_moves = [i for i, card in enumerate(env.player_hand) if card != 0]
        if not valid_moves:
            print("No valid moves left!")
            break

        # AI predicts only from valid moves
        ai_probs = model.predict(state.reshape(1, -1), verbose=0)[0]
        ai_valid_probs = [(i, ai_probs[i]) for i in valid_moves]
        ai_action = max(ai_valid_probs, key=lambda x: x[1])[0]

        action = int(input(f"Choose card position {valid_moves}: "))
        while action not in valid_moves:
            action = int(input(f"Invalid! Choose from {valid_moves}: "))

        print(f"AI would play position: {ai_action}")
        state, reward, done, _ = env.step(action)
        print(f"Reward: {reward}")

human_vs_ai()