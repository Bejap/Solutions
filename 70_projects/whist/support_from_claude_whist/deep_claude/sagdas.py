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


class EpsilonGreedyPolicy:
    def __init__(self, num_actions, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.995):
        self.num_actions = num_actions
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def get_action(self, q_values=None):
        if np.random.random() < self.epsilon:
            return np.random.random(self.num_actions)

        if q_values is None:
            return np.random.randint(self.num_actions)

        return np.argmax(q_values)

    def update(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


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


class WhistGame:
    def __init__(self, num_players=4):
        self.deck = Deck()
        self.players = [Player(str(i + 1)) for i in range(num_players)]
        self.epsilon_policy = EpsilonGreedyPolicy(num_actions=len(self.players[0].hand))

    def reset_game(self):
        # Reset deck and deal cards
        self.deck = Deck()
        self.deck.shuffle()

        # Deal cards equally to players
        cards_per_player = len(self.deck.card_deck) // len(self.players)
        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)

        return self._get_game_state()

    def _get_game_state(self):
        # Convert game state to a format suitable for AI
        return {
            'player_hands': [player.get_hand() for player in self.players],
            'current_trick': []  # Could be expanded to track trick state
        }

    def play_round(self):
        # Simple round simulation
        actions = []
        played_cards = []

        for player in self.players:
            # Get Q-values (placeholder - in real implementation, this would come from a trained model)
            q_values = np.random.rand(len(player.hand))

            # Choose action using epsilon-greedy policy
            action_index = self.epsilon_policy.get_action(q_values)

            # Play the selected card
            card = player.action(action_index)

            if card:
                actions.append(action_index)
                played_cards.append(card)

        # Update epsilon policy
        self.epsilon_policy.update()

        return {
            'actions': actions,
            'played_cards': played_cards
        }


def main():
    # Training loop
    game = WhistGame()
    num_episodes = 500

    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # Reset game for new episode
        initial_state = game.reset_game()

        # Play a round
        round_result = game.play_round()

        # In a full implementation, you'd:
        # 1. Calculate reward
        # 2. Update agent's memory
        # 3. Train the agent

        # Print episode details
        print(f"\nEpisode {episode + 1}")
        print(f"Epsilon: {game.epsilon_policy.epsilon:.4f}")
        print("Actions:", round_result['actions'])
        print("Played Cards:", round_result['played_cards'])


if __name__ == "__main__":
    main()