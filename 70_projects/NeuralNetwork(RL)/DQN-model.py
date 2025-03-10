import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque


# --- Definer kort, spiller og kortbunke ---
class Card:
    def __init__(self, suit, rank, rank_value):
        self.suit = suit
        self.rank = rank
        self.rank_value = rank_value


class Player:
    def __init__(self, is_human=False):
        self.hand = []
        self.is_human = is_human


class Deck:
    def __init__(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = list(range(2, 15))  # 2-14 (med 14 som es)
        self.cards = [Card(suit, rank, rank) for suit in suits for rank in ranks]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num):
        return [self.cards.pop() for _ in range(num)]


# --- Whist-relaterede variabler ---
players = [Player() for _ in range(4)]
deck = Deck()
deck.shuffle()
lead_player = 0
score_array = [0] * 4


# --- Deep Q-Network Model ---
def build_dqn():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(52,)),
        Dense(64, activation='relu'),
        Dense(13, activation='linear')  # 13 mulige kortvalg
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


dqn_model = build_dqn()


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


replay_buffer = ReplayBuffer()


def get_whist_state():
    """Returnerer en numerisk repræsentation af spilsituationen."""
    state = np.zeros(52)  # Placeholder, skal opdateres med relevant tilstand
    return state


def reset_whist_game():
    global players, deck, score_array, lead_player
    deck = Deck()
    deck.shuffle()
    for player in players:
        player.hand = deck.deal(13)
    score_array = [0] * 4
    lead_player = random.randint(0, 3)
    return get_whist_state()


def step_whist_game(action):
    global lead_player, score_array, players
    current_player = players[lead_player]
    if action < 0 or action >= len(current_player.hand):
        raise ValueError("Ugyldig handling: action uden for rækkevidde")
    played_card = current_player.hand.pop(action)
    lead_player = (lead_player + 1) % 4
    reward = 0
    done = False
    if lead_player == 0:
        winner_index = determine_winner()
        score_array[winner_index] += 1
        reward = 1 if winner_index == 0 else -1
    if sum(score_array) >= 13:
        done = True
    return get_whist_state(), reward, done, {}


def determine_winner():
    """Bestemmer vinderen af en runde."""
    return random.randint(0, 3)  # Placeholder-logik


def select_action(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return random.randint(0, 12)  # Tilfældig handling (udforskning)
    q_values = dqn_model.predict(np.array([state]), verbose=0)[0]
    return np.argmax(q_values)


def train_dqn(batch_size=32, gamma=0.99):
    if len(replay_buffer.buffer) < batch_size:
        return
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)
    q_values = dqn_model.predict(states, verbose=0)
    next_q_values = dqn_model.predict(next_states, verbose=0)
    for i in range(batch_size):
        target_q = rewards[i] if dones[i] else rewards[i] + gamma * np.max(next_q_values[i])
        q_values[i][actions[i]] = target_q
    dqn_model.fit(states, q_values, epochs=1, verbose=0)


def train_agent(episodes=1000, epsilon_decay=0.995, epsilon_min=0.01):
    epsilon = 1.0
    for episode in range(episodes):
        state = reset_whist_game()
        done = False
        total_reward = 0
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = step_whist_game(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            train_dqn()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % 100 == 0:
            print(f"Episode {episode}: Total reward {total_reward}, Epsilon {epsilon:.3f}")


train_agent()
