import random
import numpy as np
import tensorflow as tf
from collections import deque
from tqdm import tqdm


class Card:
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}
    SUIT_VALUES = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.rank_value = self._get_rank_value()
        self.suit_value = self._get_suit_value()

    def _get_rank_value(self):
        return self.RANK_VALUES[self.rank]

    def _get_suit_value(self):
        return self.SUIT_VALUES[self.suit]

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

class Deck:
    def __init__(self):
        self.card_deck = []
        for suit in Card.SUIT_VALUES:
            for rank in Card.RANK_VALUES:
                self.card_deck.append(Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.card_deck)

    def deal(self, num_cards: int):
        return [self.card_deck.pop() for _ in range(num_cards)]

    def get_deck(self):
        return self.card_deck

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.tricks_won = 0

    def action(self, choice, hand=None):
        if hand is None:
            hand = self.hand

        if 0 <= choice < len(hand):
            return hand.pop(choice)
        return None

    def get_hand(self):
        # Sort hand by suit and rank
        self.hand.sort(key=lambda card: (card.suit_value, card.rank_value))
        return [(card.suit_value, card.rank_value) for card in self.hand]


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Reshape state to match model input
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample from memory
        minibatch = random.sample(self.memory, batch_size)

        # Process each memory entry
        for state, action, reward, next_state, done in minibatch:
            # Reshape states
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])

            # Predict Q values
            target = reward
            if not done:
                # Use target as future reward
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))

            # Get current Q values
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the model
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class WhistGame:
    def __init__(self, num_players=4):
        self.deck = Deck()
        self.players = [Player(str(i + 1)) for i in range(num_players)]
        self.current_trick = []
        self.trick_winner = None

    def reset_game(self):
        # Reset deck and deal cards
        self.deck = Deck()
        self.deck.shuffle()

        # Deal cards equally to players
        cards_per_player = len(self.deck.card_deck) // len(self.players)
        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)

        # Reset trick and winner
        self.current_trick = []
        self.trick_winner = None

        return self._get_game_state()

    def _get_game_state(self):
        # Flatten the state representation
        state = []
        for player in self.players:
            # Add player's hand characteristics
            state.extend([
                len(player.hand),  # Number of cards
                sum(card.rank_value for card in player.hand)  # Total hand strength
            ])
        return np.array(state)

    def calculate_reward(self, player, action):
        """
        Calculate reward based on the action and game state
        """
        # Basic reward structure
        base_reward = 0

        # Reward for playing a high-value card
        if player.hand and action < len(player.hand):
            card = player.hand[action]

            # Reward for playing high-rank cards
            base_reward += card.rank_value / 14.0

            # Bonus for potentially winning the trick
            if not self.current_trick or card.rank_value > max(c.rank_value for c in self.current_trick):
                base_reward += 0.5

        # Penalty for having fewer cards (encouraging playing cards)
        base_reward -= (1 / len(player.hand)) * 0.1

        return base_reward

    def play_round(self, agent):
        # Reset game state
        initial_state = self.reset_game()
        done = False
        total_reward = 0

        # Play a full round
        while not done:
            for player in self.players:
                # Get current state
                current_state = self._get_game_state()

                # Agent chooses action
                action = agent.act(current_state)

                # Perform action
                if action < len(player.hand):
                    card = player.action(action)
                    self.current_trick.append(card)

                # Calculate reward
                reward = self.calculate_reward(player, action)
                total_reward += reward

                # Get next state
                next_state = self._get_game_state()

                # Check if round is done
                done = len(player.hand) == 0

                # Store memory for learning
                agent.remember(current_state, action, reward, next_state, done)

            # Determine trick winner (simplified)
            if self.current_trick:
                self.trick_winner = max(self.current_trick, key=lambda c: c.rank_value)
                self.current_trick = []

        return total_reward


def main():
    # Initialize game and agent
    game = WhistGame()

    # State size: 2 values per player (hand size, hand strength)
    # Action size: maximum number of cards in hand
    agent = DQNAgent(state_size=8, action_size=13)

    # Training loop
    episodes = 1000
    batch_size = 32

    for episode in tqdm(range(episodes), desc="Training"):
        # Play a round
        reward = game.play_round(agent)

        # Train the agent
        agent.replay(batch_size)

        # Print progress periodically
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}, Epsilon: {agent.epsilon:.2f}")


if __name__ == "__main__":
    main()