class Card:
    '''
    A class representing a playing card.

    Each card has two attributes:
        suit: str - The suit of the card (Hearts, Diamonds, Clubs, Spades)
        rank: str - The rank of the card (2-10, J, Q, K, A)

    The card has no inherent value - the game rules determine
    how ranks are interpreted and compared.

    Example:
        card = Card("Hearts", "A")  # Creates the Ace of Hearts
    '''
    # Define rank values as a class variable
    RANK_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14
    }

    SUIT_VALUES = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.short_rank = self._get_short_rank()
        self.short_suit = suit[0].upper()
        self.rank_value = self._get_rank_value()
        self.suit_value = self._get_suit_value()

    def _get_rank_value(self):
        """Convert rank to numerical value for comparison."""
        return self.RANK_VALUES[self.rank]

    def _get_suit_value(self):
        """Convert suit to numerical value for sorting."""
        return self.SUIT_VALUES[self.suit]

    def _get_short_rank(self):
        if self.rank == '10':
            return '10'
        return self.rank[0].upper()

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def get_short_name(self):
        return f"{self.short_rank}o{self.short_suit}"

    def __repr__(self):
        return self.get_short_name()

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.suit == other.suit and self.rank == other.rank
        return False

    def __lt__(self, other):
        """Enable sorting of cards."""
        if self.suit_value != other.suit_value:
            return self.suit_value < other.suit_value
        return self.rank_value < other.rank_value