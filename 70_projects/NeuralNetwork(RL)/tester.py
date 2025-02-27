import random


class Card:
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}
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
        self.cards = [Card(suit, rank) for suit in Card.SUIT_VALUES.keys()
                      for rank in Card.RANK_VALUES.keys()]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand = []

    def __str__(self):
        return f"Player {self.player_id}: {', '.join(str(card) for card in self.hand)}"

    def play_card(self, lead_suit):
        valid_cards = [card for card in self.hand if card.suit_value == lead_suit]
        if valid_cards:
            card_to_play = valid_cards.pop(0)
        else:
            card_to_play = self.hand.pop(0)
        return card_to_play


# Initialisering
players = [Player(i) for i in range(4)]
deck = Deck()
deck.shuffle()


score_array = [0] * 4
trump_suit = random.randint(0, 5)
print(f"Trump suit: {trump_suit}\n")

cards_array = [0] * 52
round_array = [0] * 52
hand_arrays = [[0] * 52 for _ in range(4)]
lead_array = [0] * 4
trump_array = [0] * 5
player_array = [0] * 4
trump_array[trump_suit] = 1


for player in players:
    player.hand = deck.deal(13)
    for card in player.hand:
        card_position = (card.suit_value - 1) * 13 + (card.rank_value - 2)
        hand_arrays[player.player_id][card_position] = 1

def update_hand_state(player_index, card):
    card_position = (card.suit_value - 1) * 13 + (card.rank_value - 2)
    hand_arrays[player_index][card_position] = 0


def determine_winner(played_cards, lead_suit, trump_suit):
    lead_cards = [card for card in played_cards if card.suit == lead_suit]
    trump_cards = [card for card in played_cards if card.suit == trump_suit]
    if trump_cards:
        return played_cards.index(max(trump_cards, key=lambda card: card.rank_value))
    return played_cards.index(max(lead_cards, key=lambda card: card.rank_value))


# Spil 13 runder
lead_player = 0
for round_num in range(13):
    print(f"Round {round_num + 1}:")
    played_cards = []
    lead_suit = None
    round_array = [0] * 52
    lead_array = [0] * 4

    for i in range(4):
        current_player = (lead_player + i) % 4
        if i == 0:
            played_card = players[current_player].play_card(None)
            lead_suit = played_card.suit
            lead_array[Card.SUIT_VALUES[lead_suit] - 1] = 1
        else:
            played_card = players[current_player].play_card(lead_suit)

        played_cards.append(played_card)
        update_hand_state(current_player, played_card)

        card_position = (played_card.suit_value - 1) * 13 + (played_card.rank_value - 2)
        round_array[card_position] = 1
        cards_array[card_position] = 1
        player_array[current_player] = 1

        print(f"Player {current_player + 1} played {played_card}")

    winner_index = determine_winner(played_cards, lead_suit, trump_suit)
    winner_player = (lead_player + winner_index) % 4
    score_array[winner_player] += 1
    lead_player = winner_player

    print(f"Winner of the round: Player {winner_player + 1}\n")
    print(f"Game State:\nCards Played: {cards_array}\nRound State: {round_array}\nHand Array Player 1: {hand_arrays[0]}\nHand Array Player 2: {hand_arrays[1]}\nHand Array Player 3: {hand_arrays[2]}\nHand Array Player 4: {hand_arrays[3]}\nLead Suit: {lead_array}\nTrump Suit: {trump_array}\nActive Player: {player_array}\n")

print(f"Final Scores: {score_array}")
