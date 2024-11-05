import random
import Card as card

class Deck:
    def __init__(self):
        self.cards: list = []
        suits: list = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks: list = ['2', '3', '4', '5', '6', '7', '8', '9', '10',
                       'Jack', 'Queen', 'King', 'Ace']
        for suit in suits:
            for rank in ranks:
                self.cards.append(card.Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]

