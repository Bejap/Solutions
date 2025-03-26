import random


class Card:
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14
                   }

    SUIT_VALUES = {'Hearts': 0}

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.rank_value = self._get_rank_value()
        self.suit_value = self._get_suit_value()

    def _get_rank_value(self):
        """Convert rank to numerical value for comparison."""
        return self.RANK_VALUES[self.rank]

    def _get_suit_value(self):
        """Convert suit to numerical value for sorting."""
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
        self.last_played_card = None
        self.name = name
        self.hand = []
        self.tricks_won = 0

    def action(self, choice):
        self._sort_hand()
        if 0 <= choice < len(self.hand):
            return self.hand[choice]

        # self.last_played_card = self.hand[choice]

    def _sort_hand(self):
        self.hand.sort(key=lambda card: (card.suit_value, card.rank_value))

    def get_hand(self):
        self._sort_hand()  # Sort the hand first
        return [(card.suit_value, card.rank_value) for card in self.hand]

    def __repr__(self):
        return f"Player {self.name}, Hand: {self.get_hand()}"
