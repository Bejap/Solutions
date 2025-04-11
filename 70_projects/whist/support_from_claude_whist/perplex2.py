import random


class Card:
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
                   }

    SUIT_VALUES = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}

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
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        for suit in suits:
            for rank in Card.RANK_VALUES:
                self.card_deck.append(Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.card_deck)

    def deal(self, num_cards: int):
        return [self.card_deck.pop() for _ in range(num_cards)]

    def get_deck(self):
        return self.card_deck


class Player:
    def __init__(self, id):
        self.last_played_card = None
        self.hand = []
        self.tricks_won = 0
        self.id = id
        self.known_actions = []

    def action(self, choice):
        self._sort_hand()
        if 0 <= choice < len(self.hand):  # Fixed off-by-one error
            played_card = self.hand.pop(choice)  # Actually remove card
            self.last_played_card = played_card
            return played_card
        raise IndexError("Invalid card choice")

        # self.last_played_card = self.hand[choice]

    def observe(self, player_id, action):
        if player_id != self.id:
            self.known_actions.append((player_id, action))

    def return_other_hand(self, hand, list_length):
        player_card_list = [[0] * list_length for _ in range(4)]
        # print(self.id)
        player_card_list[self.id - 1] = hand
        print("Player ", self.id,"this is the hand: ", hand)
        print("Player ", self.id, "knows ", self.known_actions)

        for p_id, act in self.known_actions:
            card_pos = act - 2
            player_card_list[p_id - 1][card_pos] = 1

        print(player_card_list[0], player_card_list[1], player_card_list[2], player_card_list[3], "\n")

        return player_card_list[0], player_card_list[1], player_card_list[2], player_card_list[3]

    def resetting_observation(self):
        self.known_actions = []

    def _sort_hand(self):
        self.hand.sort(key=lambda card: (card.suit_value, card.rank_value))

    def get_hand(self):
        self._sort_hand()  # Sort the hand first
        return [(card.suit_value, card.rank_value) for card in self.hand]

    def __repr__(self):
        return f"Player {self.id}, Hand: {self.get_hand()}"
