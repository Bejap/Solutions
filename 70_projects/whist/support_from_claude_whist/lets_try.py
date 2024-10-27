import random
import tensorflow as tf
import numpy as np
from collections import deque
import time
import os
import json


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
        if self.rank=='10':
            return f"{self.rank}{self.suit[0]}"
        else:
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


class WhistStateEncoder:
    def __init__(self):
        # Initialize encoding values for cards
        self.card_encoding = {f"{rank} of {suit}": i for i, (rank, suit) in
                              enumerate((rank, suit) for suit in Card.SUIT_VALUES for rank in Card.RANK_VALUES)}

    def encode_state(self, player_hand, played_cards, current_trick, trump_suit):
        # Initialize a zeroed state vector of fixed size (312 here, as per the network input requirement)
        state = np.zeros(312, dtype=np.float32)

        # Encode the player's hand (for instance, in the first 52 indices)
        for card in player_hand:
            card_index = self.card_encoding.get(f"{card.rank} of {card.suit}")
            if card_index is not None:
                state[card_index] = 1

        # Encode the played cards (for example, next 52 indices)
        for card in played_cards:
            card_index = self.card_encoding.get(f"{card.rank} of {card.suit}")
            if card_index is not None:
                state[52 + card_index] = 1

        # Encode the trump suit as a one-hot encoding (next 4 indices)
        suit_index = Card.SUIT_VALUES.get(trump_suit)
        if suit_index is not None:
            state[104 + suit_index] = 1

        # Additional features like the current trick could go here as needed

        return state

    def create_card_encoding(self):
        # Creates a dictionary that maps each card to a unique integer
        suits = ['spades', 'hearts', 'diamonds', 'clubs']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

        # Generate a unique integer for each card, with 52 unique entries
        card_encoding = {f"{rank}_of_{suit}": i for i, (rank, suit) in enumerate((rank, suit) for rank in ranks for suit in suits)}
        return card_encoding

    def encode(self, game_state):
        # Initialize a zero vector of the fixed size
        state_vector = [0] * 312

        # Encode each player's hand, assigning a unique index to each card
        for player_idx, hand in enumerate(game_state['hands']):
            for card in hand:
                card_index = self.card_encoding[card]
                # Example: first 52 indices are Player 1's hand, next 52 for Player 2, etc.
                state_vector[player_idx * 52 + card_index] = 1

        # Encode cards currently on the table (the trick)
        for card in game_state['current_trick']:
            card_index = self.card_encoding[card]
            # Example: indices 208-259 for current trick
            state_vector[208 + card_index] = 1

        # Additional features could include which player's turn it is, or points scored
        # Player turn encoding (using last 4 indices, one per player)
        state_vector[260 + game_state['current_player']] = 1

        return state_vector

@tf.keras.utils.register_keras_serializable()
class WhistModel(tf.keras.Model):
    def __init__(self):
        super(WhistModel, self).__init__()
        # Define layers
        self.layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.layer3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(52)

        # Build the model to initialize weights
        self.build((None, 312))  # Initialize with an input shape of 312

    def build(self, input_shape):
        # Build each layer with correct input shapes
        self.layer1.build(input_shape)
        layer1_output_shape = (input_shape[0], 256)
        self.layer2.build(layer1_output_shape)
        layer2_output_shape = (input_shape[0], 128)
        self.layer3.build(layer2_output_shape)
        layer3_output_shape = (input_shape[0], 64)
        self.output_layer.build(layer3_output_shape)
        super(WhistModel, self).build(input_shape)

    def call(self, inputs):
        # Define the forward pass
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output_layer(x)

    def get_config(self):
        """
        Returns a dictionary containing the configuration of the model.
        This config will be used for saving and loading the model.
        """
        config = super(WhistModel, self).get_config()  # Get base config from tf.keras.Model
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of WhistModel from the configuration dictionary.
        This is used when loading a saved model.
        """
        return cls()  # Return a new instance since there are no custom arguments



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

    @tf.function(reduce_retracing=True)
    def get_action(self, state, legal_actions, training=True):
        """Choose action using epsilon-greedy policy, ensuring legality."""
        # Convert inputs to tensors with specific shapes
        state = tf.ensure_shape(
            tf.convert_to_tensor(state, dtype=tf.float32),
            [312]
        )
        legal_actions = tf.convert_to_tensor(legal_actions, dtype=tf.int32)

        if training and tf.random.uniform((), dtype=tf.float32) < self.epsilon:
            random_idx = tf.random.uniform((), maxval=tf.shape(legal_actions)[0], dtype=tf.int32)
            return tf.gather(legal_actions, random_idx)

        # Get Q-values for all actions
        state_tensor = tf.expand_dims(state, 0)
        q_values = self.model(state_tensor)[0]

        # Create mask for legal actions
        mask = tf.fill(q_values.shape, tf.cast(-float('inf'), tf.float32))
        legal_indices = tf.expand_dims(legal_actions, 1)
        updates = tf.ones(tf.shape(legal_actions), dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, legal_indices, updates)

        # Apply mask and get best legal action
        masked_q_values = q_values + mask
        return tf.argmax(masked_q_values, output_type=tf.int32)

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
        states = tf.convert_to_tensor([x[0] for x in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([x[1] for x in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([x[2] for x in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([x[3] for x in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([x[4] for x in batch], dtype=tf.float32)

        loss = self.train_step(states, actions, rewards, next_states, dones)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


class Whist:
    def __init__(self, player_names, human_player_index=0):
        self.players = [
            Player(name, is_human=(i == human_player_index)) for i, name in enumerate(player_names)
        ]
        self.deck = Deck()
        self.trump_suit = None

    def deal_cards(self):
        self.deck = Deck()  # Create a new deck for each game
        self.deck.shuffle()
        for player in self.players:
            player.hand = self.deck.deal(13)
            player.sort_hand()
            player.tricks_won = 0  # Reset tricks won
        self.trump_suit = random.choice(list(Card.SUIT_VALUES.keys()))
        print(self.trump_suit)

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
                print(trick_cards)
                print(f"{current_player.name}, it's your turn.")
                print(f"Your hand: {', '.join(str(card) for card in current_player.hand)}")
                print(f"Playable cards: {', '.join(str(card) for card in playable_cards)}")

                # Let the human player select a card from their playable cards
                while True:
                    try:
                        choice = int(input("Select the index of the card you want to play: "))
                        played_card = playable_cards[choice]
                        break
                    except (ValueError, IndexError):
                        print("Invalid selection. Please select a valid index.")
            else:
                # For AI players, use the AI model to choose the action
                state = ai.encoder.encode_state(current_player.hand, trick_cards, None, self.trump_suit)
                legal_actions = [i for i, card in enumerate(current_player.hand) if card in playable_cards]
                action = ai.get_action(state, legal_actions, training=True)
                played_card = current_player.hand[action]

            trick_cards.append(played_card)
            current_player.hand.remove(played_card)


            if not lead_suit:
                lead_suit = played_card.suit

            if not winning_card:
                winning_card = played_card
                winning_player = current_player
            elif played_card.suit == self.trump_suit and winning_card.suit != self.trump_suit:
                winning_card = played_card
                winning_player = current_player
            elif played_card.suit == winning_card.suit and played_card.rank_value > winning_card.rank_value:
                winning_card = played_card
                winning_player = current_player
        print(trick_cards)
        winning_player.tricks_won += 1
        return self.players.index(winning_player)

    def play_game(self, ai):
        self.deal_cards()
        leading_player_index = 0

        for _ in range(13):  # 13 tricks per game
            leading_player_index = self.play_trick(leading_player_index, ai)

        winner = max(self.players, key=lambda x: x.tricks_won)
        print(f"The winner is {winner.name} with {winner.tricks_won} tricks won!")
        return winner


def train_ai():
    ai = WhistAI()
    for episode in range(1000):
        game = Whist(["Player1", "AI1", "AI2", "AI3"], human_player_index=0)
        winner = game.play_game(ai)

        if episode % 100 == 0:
            ai.update_target_network()
            print(f"Episode {episode} completed. Epsilon: {ai.epsilon:.4f}")
    ai.model.save('../claude(need_debug).keras')


if __name__ == "__main__":
    train_ai()