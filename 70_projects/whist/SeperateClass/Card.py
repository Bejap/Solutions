class Card:
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f'{self.suit} of {self.rank}'

    def _get_short_rank(self):
        if self.rank == '10':
            return '10'
        return self.rank[0].upper()

    def _get_short_suit(self):
        return self.suit[0].upper()

    def get_short_name(self):
        return f"{self._get_short_rank()}o{self._get_short_suit()}"