import random
import tensorflow as tf
import numpy as np
from collections import deque
import time


class Card:
    # Define rank values and suits as a class variable
    RANK_VALUES = {str(num): num for num in range(2, 11)}
    RANK_VALUES.update({'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14})
    SUIT_VALUES = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.rank_value = self.RANK_VALUES[self.rank]
        self.suit_value = self.SUIT_VALUES[self.suit]

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __repr__(self):
        return f"{self.rank[0]}o{self.suit[0]}"

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __lt__(self, other):
        if self.suit_value != other.suit_value:
            return self.suit_value < other.suit_value
        return self.rank_value < other.rank_value


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Card.SUIT_VALUES for rank in Card.RANK_VALUES]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]


class Player:
    def __init__(self, name, is_human=False):
        self.name = name
        self.hand = []
        self.tricks_won = 0
        self.is_human = is_human
        self.cards_seen = set()

    def sort_hand(self):
        self.hand.sort()

    def play_card(self, playable_cards):
        card = playable_cards[0]
        self.hand.remove(card)
        return card


class WhistStateEncoder:
    def encode_state(self, player_hand, played_cards, current_trick, trump_suit):
        state = np.zeros(312)
        return state.astype(np.float32)


class WhistModel(tf.keras.Model):
    def __init__(self):
        super(WhistModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.layer3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(52)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output_layer(x)


class WhistAI:
    def __init__(self):
        self.model = WhistModel()
        self.target_model = WhistModel()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.encoder = WhistStateEncoder()
        self.update_target_network()

    @tf.function
    def get_action(self, state, legal_actions, training=True):
        """Choose action using epsilon-greedy policy, ensuring legality."""
        if training and tf.random.uniform(()) < self.epsilon:
            # Random legal action
            return tf.convert_to_tensor(random.choice(legal_actions))

        # Get Q-values for all actions
        state_tensor = tf.expand_dims(state, 0)
        q_values = self.model(state_tensor)[0]

        # Mask illegal actions by setting them to a very low value
        mask = tf.fill(q_values.shape, -float('inf'))
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(legal_actions, axis=1), tf.ones(len(legal_actions)))
        legal_q_values = q_values + mask

        # Select the action with the highest Q-value among legal actions
        best_action = tf.argmax(legal_q_values)

        return best_action

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            current_q = self.model(states)
            current_q_values = tf.gather(current_q, actions, batch_dims=1)
            next_q = self.target_model(next_states)
            max_next_q = tf.reduce_max(next_q, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
            loss = self.loss_fn(target_q_values, current_q_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        loss = self.train_step(states, actions, rewards, next_states, dones)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


class Whist:
    def __init__(self, player_names, human_player_index=0):
        self.players = [Player(name, is_human=(i == human_player_index)) for i, name in enumerate(player_names)]
        self.deck = Deck()
        self.trump_suit = None

    def deal_cards(self):
        self.deck.shuffle()
        for player in self.players:
            player.hand = self.deck.deal(13)
            player.sort_hand()
        self.trump_suit = random.choice(['Hearts', 'Diamonds', 'Clubs', 'Spades'])

    def play_trick(self, leading_player_index, ai):
        trick_cards = []
        lead_suit = None
        winning_card = None
        winning_player = None

        for i in range(len(self.players)):
            current_player = self.players[(leading_player_index + i) % len(self.players)]
            playable_cards = [card for card in current_player.hand if card.suit == lead_suit] if lead_suit else current_player.hand
            if not playable_cards:
                playable_cards = current_player.hand

            if current_player.is_human:
                played_card = playable_cards[0]
            else:
                state = ai.encoder.encode_state(current_player.hand, trick_cards, None, self.trump_suit)
                action = ai.get_action(state, [i for i, card in enumerate(current_player.hand)], training=True)
                played_card = current_player.hand[action]

            trick_cards.append(played_card)
            current_player.hand.remove(played_card)

            if not lead_suit:
                lead_suit = played_card.suit

            if not winning_card or (played_card.suit == winning_card.suit and played_card.rank_value > winning_card.rank_value):
                winning_card = played_card
                winning_player = current_player

        winning_player.tricks_won += 1
        return self.players.index(winning_player)

    def play_game(self, ai):
        self.deal_cards()
        leading_player_index = 0

        for _ in range(13):
            leading_player_index = self.play_trick(leading_player_index, ai)

        winner = max(self.players, key=lambda x: x.tricks_won)
        return winner


def train_ai():
    ai = WhistAI()
    for episode in range(1000):
        game = Whist(["You", "AI1", "AI2", "AI3"], human_player_index=0)
        winner = game.play_game(ai)

        if episode % 100 == 0:
            ai.update_target_network()
            print(f"Episode {episode} completed.")

if __name__ == "__main__":
    train_ai()
