import random


class Card:
    RANK_VALUES = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14
    }

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
        self.cards: list = [Card(suit, rank) for suit in ['Clubs', 'Diamonds', 'Hearts', 'Spades']
                            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10',
                                         'Jack', 'Queen', 'King', 'Ace']]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]

    def show(self):
        return [str(card) for card in self.cards]


class Player:
    def __init__(self):
        self.hand = []

    def __str__(self):
        return ", ".join(str(card) for _ in self.hand)


deck = Deck()
deck.shuffle()
print("Deck after shuffling:", deck.show())
print(len(deck.cards))

cards_array = [0] * 52
round_array = [0] * 52
hand_arrays = [[0] * 52 for _ in range(4)]
lead_array = [0] * 4
trump_array = [0] * 5
player_array = [0] * 4
score_array = [0] * 4
count = 0


def cards_played(card_s):
    card_position_s = (card_s.suit_value - 1) * 13 + (card_s.rank_value - 2)
    cards_array[card_position_s] = 1
    return cards_array


def cards_in_round(card_a, count_s):
    global round_array
    if count_s % 4 == 0:
        round_array = [0] * 52

    card_position_a = (card_a.suit_value - 1) * 13 + (card_a.rank_value - 2)
    round_array[card_position_a] = 1
    return round_array


def deal_cards():
    players_a = [Player() for _ in range(4)]
    for play in players_a:
        play.hand = deck.deal(13)
    return players_a


def player_hand_state():
    hand_array = [0] * 52
    for card_l in player.hand:
        card_position_l = (card_l.suit_value - 1) * 13 + (card_l.rank_value - 2)
        hand_array[card_position_l] = 1
    return hand_array


def update_hand_state(player_index_j, card_j):
    card_position_j = (card_j.suit_value - 1) * 13 + (card_j.rank_value - 2)
    hand_arrays[player_index_j][card_position_j] = 1

def update_score(winner_index):
    score_array[winner_index] = 1


players = deal_cards()

for i, player in enumerate(players):
    for card in player.hand:
        update_hand_state(i, card)
    print(f"Player {i + 1}: {[str(card) for card in player.hand]}")
    print(f"Hand State: {hand_arrays[i]}\n")


played_cards = []
whist_state = []
lead_suit = None
trump = random.randint(0, 4)
trump_array[trump] = 1
ild = 0

for i in range(13):
    for player_index, player in enumerate(players):
        lead_array = [0] * 4
        player_array = [0] * 4
        if player.hand:
            card = None
            # card = player.hand.pop(0)
            player_array[player_index] = 1
            # print(card, "\n")

            if player_index == 0:
                card = player.hand.pop(0)
                lead_suit = card.suit_value
            if player_index > 0:
                if any(card.suit == lead_suit for card in player.hand):
                    valid_cards = [card for card in player.hand if card.suit_value == lead_suit]
                    card = valid_cards.pop(0)
                    print(card.suit)
                else:
                    card = player.hand.pop(0)
                    print(card.suit)
            played_cards.append(card)

            cards_played_state = cards_played(card)
            cards_round_state = cards_in_round(card, count)
            card_position = (card.suit_value - 1) * 13 + (card.rank_value - 2)
            hand_arrays[player_index][card_position] = 0
            lead_array = [0] * 4
            if lead_suit:
                lead_array[lead_suit - 1] = 1

            # print(f"Cards Played:       {cards_played_state}")
            # print(f"Round State:        {cards_round_state}")
            # print(f"Updated Hand State: {hand_arrays}\n")
            whist_state = [cards_played_state,
                           cards_round_state,
                           hand_arrays[player_index],
                           lead_array,
                           trump_array,
                           player_array]

            count += 1

        print(f"This is the whist state: \n{whist_state}")
