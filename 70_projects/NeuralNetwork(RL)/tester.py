import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

class Card:
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}
    SUIT_VALUES = {'Clubs': 1, 'Diamonds': 2, 'Hearts': 3, 'Spades': 4}

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.rank_value = self.RANK_VALUES[self.rank]
        self.suit_value = self.SUIT_VALUES[self.suit]

    def __str__(self):
        return f"{self.rank} of {self.suit}"


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Card.SUIT_VALUES.keys()
                      for rank in Card.RANK_VALUES.keys()]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]


class Agent:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand = []

    def __str__(self):
        return f"Player {self.player_id}: {', '.join(str(card) for card in self.hand)}"

    def play_card(self, lead_suit):
        valid_cards = [card for card in self.hand if card.suit_value == lead_suit]
        if valid_cards:
            card_to_play = valid_cards.pop(0)
        else:
            card_to_play = self.hand.pop(0)
        return card_to_play


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int = 32):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Diskonteringsfaktor
        self.epsilon = 1.0  # Udforskningsrate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Udforskning
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)  # Udnyt bedste kendte handling

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))[0]
            target_f[action] = target
            self.model.fit(np.array([state]), np.array([target_f]), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Initialisering
players = [Agent(i) for i in range(4)]
deck = Deck()
deck.shuffle()

score_array = [0] * 4
trump_suit = random.randint(0, 4)
print(f"Trump suit: {trump_suit}\n")

cards_array = [0] * 52
round_array = [0] * 52
hand_arrays = [[0] * 52 for _ in range(4)]
lead_array = [0] * 4
trump_array = [0] * 5
player_array = [0] * 4
trump_array[trump_suit] = 1

for player in players:
    player.hand = deck.deal(13)
    for card in player.hand:
        card_position = (card.suit_value - 1) * 13 + (card.rank_value - 2)
        hand_arrays[player.player_id][card_position] = 1


def update_hand_state(player_index, card):
    card_position = (card.suit_value - 1) * 13 + (card.rank_value - 2)
    hand_arrays[player_index][card_position] = 0


def determine_winner(played_cards, lead_suit, trump_suit):
    lead_cards = [card for card in played_cards if card.suit == lead_suit]
    trump_cards = [card for card in played_cards if card.suit == trump_suit]
    if trump_cards:
        return played_cards.index(max(trump_cards, key=lambda card: card.rank_value))
    return played_cards.index(max(lead_cards, key=lambda card: card.rank_value))


def build_dqn_model(input_shape, num_actions):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def select_action(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return random.randint(0, 12)
    q_values = dqn_model.predict(np.array([state]), verbose=0)
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
        target_q = rewards[i] + (1 - dones[i]) * gamma * np.max(next_q_values[i])
        q_values[i][actions[i]] = target_q

    dqn_model.fit(states, q_values, epochs=1, verbose=0)

def step_whist_game(action):
    global lead_player, score_array, lead_suit, played_cards

    # Oversæt action til et kort
    current_player = players[lead_player]
    played_card = current_player.hand.pop(action)

    # Opdater spilhistorik og fjern kort fra hånden
    update_hand_state(lead_player, played_card)

    # Gem kortet i spiltilstand
    cards_array[(played_card.suit_value - 1) * 13 + (played_card.rank_value - 2)] = 1

    # Næste spiller
    lead_player = (lead_player + 1) % 4

    # Beregn belønning (f.eks. +1 for at vinde en runde)
    reward = 0
    if lead_player == 0:  # Hvis vi har spillet en fuld runde
        winner_index = determine_winner(played_card, lead_suit, trump_suit)
        winner_player = (lead_player + winner_index) % 4
        score_array[winner_player] += 1
        reward = 1 if winner_player == 0 else -1  # AI får positiv belønning for at vinde

    # Tjek om spillet er slut
    done = sum(score_array) >= 13  # Når alle kort er spillet

    # Returnér ny tilstand, belønning, og om spillet er slut
    whist_state = get_whist_state()
    return whist_state, reward, done, {}





def get_whist_state():
    state = cards_array + round_array + hand_arrays[0] + hand_arrays[1] + hand_arrays[2] + hand_arrays[3] + lead_array + trump_array + player_array + score_array
    return np.array(state)



def reset_whist_game():
    global players, deck, score_array, lead_player

    deck = Deck()
    deck.shuffle()

    for player in players:
        player.hand = deck.deal(13)

    score_array = [0] * 4

    lead_player = random.randint(0, 3)

    whist_state = get_whist_state()
    return whist_state


input_shape = 329
num_actions = 13
dqn_model = build_dqn_model(input_shape, num_actions)
episodes = 10000
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
replay_buffer = ReplayBuffer(max_size=10000)  # Create an instance

for episode in range(episodes):
    state = reset_whist_game()
    done = False

    while not done:
        action = select_action(state)
        next_state, reward, done, _ = step_whist_game(action) # Corrected unpacking
        replay_buffer.add(state, action, reward, next_state, done)  # Use the instance
        train_dqn()

        state = next_state

    epsilon = max(epsilon_min, epsilon_decay * epsilon) #Corrected decay

    if episode % 100 == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon}")



lead_player = 0
for round_num in range(13):
    print(f"Round {round_num + 1}:")
    played_cards = []
    lead_suit = None
    round_array = [0] * 52
    lead_array = [0] * 4

    for i in range(4):
        current_player = (lead_player + i) % 4
        if i == 0:
            played_card = players[current_player].play_card(None)
            lead_suit = played_card.suit
            lead_array[Card.SUIT_VALUES[lead_suit] - 1] = 1
        else:
            played_card = players[current_player].play_card(lead_suit)

        played_cards.append(played_card)
        update_hand_state(current_player, played_card)

        card_position = (played_card.suit_value - 1) * 13 + (played_card.rank_value - 2)
        round_array[card_position] = 1
        cards_array[card_position] = 1
        player_array[current_player] = 1

        print(f"Player {current_player + 1} played {played_card}")

    winner_index = determine_winner(played_cards, lead_suit, trump_suit)
    winner_player = (lead_player + winner_index) % 4
    score_array[winner_player] += 1
    lead_player = winner_player

    print(f"Winner of the round: Player {winner_player + 1}\n")
    print(f"Game State:\nCards Played:  {cards_array}"
          f"\nRound State:  {round_array}"
          f"\nHand Array Player 1: {hand_arrays[0]}"
          f"\nHand Array Player 2: {hand_arrays[1]}"
          f"\nHand Array Player 3: {hand_arrays[2]}"
          f"\nHand Array Player 4: {hand_arrays[3]}"
          f"\nLead Suit: {lead_array}"
          f"\nTrump Suit: {trump_array}"
          f"\nActive Player: {player_array}\n")

    whist_state = cards_array + round_array + hand_arrays[0] + hand_arrays[1] + hand_arrays[2] + hand_arrays[3] + lead_array + trump_array + player_array + score_array
    print(f"\nThis is the Whist state {whist_state}")

replay_buffer = ReplayBuffer(max_size=10000)
print(f"Final Scores: {score_array}")
