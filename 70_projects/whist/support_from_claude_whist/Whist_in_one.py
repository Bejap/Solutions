import random
import numpy as np
from StateEncoder import create_state_vector

class Card:
    """
    A class representing a playing card.

    Each card has two attributes:
        suit: str - The suit of the card (Hearts, Diamonds, Clubs, Spades)
        rank: str - The rank of the card (2-10, J, Q, K, A)

    The card has no inherent value - the game rules determine
    how ranks are interpreted and compared.

    Example:
        card = Card("Hearts", "A")  # Creates the Ace of Hearts
    """
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


class Deck:
    def __init__(self):
        self.cards: list = []
        suits: list = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks: list = ['2', '3', '4', '5', '6', '7', '8', '9', '10',
                       'Jack', 'Queen', 'King', 'Ace']
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]


class Player:

    def __init__(self, name: str, is_human: bool, team: int):
        self.name = name
        self.hand = []
        self.tricks_won = 0
        self.is_human = is_human
        self.cards_seen = set()
        self.team = team
        self.passed = False
        self.bid = 0
        self.points = 0

    def play_card(self, playable_cards: list[Card], trump_suit: str, lead_suit=None, played_cards=None):
        playable_cards.sort()

        while True:
            print(f"\nYour hand: {[card.get_short_name() for card in sorted(self.hand)]}")
            print(f"Playable cards: {[card.get_short_name() for card in playable_cards]}")
            if lead_suit:
                print(f"Lead suit: {lead_suit}")

            try:
                card_idx = int(input(f"Enter the number (1-{len(playable_cards)}) of the card you want to play: ")) - 1
                if 0 <= card_idx < len(playable_cards):
                    card = playable_cards[card_idx]
                    self.hand.remove(card)  # Remove card from hand
                    print(f"You played: {card.get_short_name()}")
                    return card
                else:
                    print("Invalid card number. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def sort_hand(self):
        self.hand.sort()

    def make_bid(self, current_highest_bid):
        print(f"\nYour hand: {[card.get_short_name() for card in sorted(self.hand)]}")
        if self.passed:
            print(f"{self.name} has already passed.")
            return self.bid

        while True:
            action = int(input(f"{self.name}, do you want to bid or pass? (1/0): "))
            if not action:
                self.passed = True
                print(f"{self.name} has passed.")
                return None
            elif action:
                try:
                    bid = int(input(f"Enter your bid (must be higher than {current_highest_bid}): "))
                    if current_highest_bid < bid <= 13:
                        self.bid = bid
                        return bid
                    else:
                        print(f"Bid must be higher than the current highest bid ({current_highest_bid}, but max 13).")
                except ValueError:
                    print("Please enter a valid number.")
            else:
                print("Invalid action. Please type 'bid' or 'pass'.")


class Whist:
    def __init__(self, player_names: list[str]):
        # Assign players to teams
        self.cards_played = None
        self.highest_bidder = None
        self.highest_bid = None
        self.active_players = None
        self.deck = None
        self.players = [Player(name, is_human=False, team=i % 2) for i, name in enumerate(player_names)]
        self.count = 0
        self.trump_opti = ["Hearts", "Spades", "Diamonds", "Clubs", None]
        self.team_scores = {0: 0, 1: 0}
        self.winning_score = 0
        self.winning_team = None
        self.base_scores = {7: 1, 8: 3, 9: 6, 10: 10, 11: 15, 12: 22, 13: 30}
        self.file_name = "WhistData"
        self.cards_played = []

    def deal_cards(self):
        self.deck.shuffle()
        cards_per_player = 13  # CHANGE THIS
        self.count += 1

        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)
            player.sort_hand()

    def get_state(self, current_player_index: int):
        current_player = self.players[current_player_index]
        lead_suit = self.cards_played[0].suit if self.cards_played else None

        return create_state_vector(
            current_player=current_player,
            trick_cards=self.cards_played,
            trump_suit=self.trump,
            highest_bid=self.highest_bid,
            highest_bidder=self.highest_bidder,
            lead_suit=lead_suit
        )

    def calculate_score(self, bid, tricks_won):
        base_score = self.base_scores.get(bid, 0)
        if base_score <= tricks_won:
            extra_tricks = max(tricks_won - bid, 0)  # If tricks_won > bid, extra tricks count
            score = base_score + extra_tricks  # Add extra tricks to the base score
            return score
        else:
            print(f"You lost the round, and missed {bid - tricks_won} tricks")
            return False

    def start_bidding(self):
        print("Starting the bidding process...")
        self.highest_bid = 6
        self.highest_bidder = None
        self.active_players = len(self.players)
        while self.active_players > 1:  # Continue until only one player hasn't passed
            for player in self.players:
                if player.passed:
                    continue

                bid = player.make_bid(self.highest_bid)
                if bid is not None and bid > self.highest_bid:
                    self.highest_bid = bid
                    self.highest_bidder = player

                # Check if all players except one have passed
                self.active_players = sum(not p.passed for p in self.players)
                if self.active_players <= 1:
                    break

        # Handle the case where all but one player have passed
        if self.active_players == 1:
            self.highest_bidder = next(p for p in self.players if not p.passed)

        if self.highest_bidder:
            print(f"Bidding ended. The winner is {self.highest_bidder.name} with a bid of {self.highest_bid}.")
            return self.highest_bidder
        else:
            print("No bids were placed. No winner.")

    def play_trick(self, leading_player_index: int):
        trick_cards = []
        lead_suit = None
        winning_card = None
        winning_player = None
        self.cards_played = []

        print(f"\n{'=' * 50}")
        print(f"--- New Trick ---")
        print(f"Trump suit: {self.trump}")

        for i in range(len(self.players)):
            current_player = self.players[(leading_player_index + i) % len(self.players)]
            print(f"\n{current_player.name}'s turn:")

            playable_cards = [card for card in current_player.hand if card.suit == lead_suit] if lead_suit else current_player.hand
            if not playable_cards:
                playable_cards = current_player.hand

            played_card = current_player.play_card(playable_cards, self.trump, lead_suit, trick_cards)
            trick_cards.append(played_card)

            for player in self.players:
                player.cards_seen.add(str(played_card))

            if not lead_suit:
                lead_suit = played_card.suit
                print(f"Lead suit: {lead_suit}")

            if not winning_card:
                winning_card = played_card
                winning_player = current_player
            else:
                if (played_card.suit == winning_card.suit and played_card.rank_value > winning_card.rank_value) or \
                        (played_card.suit == self.trump and winning_card.suit != self.trump):
                    winning_card = played_card
                    winning_player = current_player
            self.cards_played.append(played_card)

        winning_player.tricks_won += 1
        self.team_scores[winning_player.team] += 1
        print(f"\n{winning_player.name} wins the trick for Team {winning_player.team + 1}!")
        return self.players.index(winning_player)

    def __play_round(self):
        s = self.start_bidding()
        print(s.name, "won, this is your hand", s.hand)
        while True:
            try:
                # Ask for user input
                j = int(input("Choose a trump suit: Hearts, Spades, Diamonds, Clubs, or None (1, 2, 3, 4, 5):\n"))
                if j not in range(5):
                    raise ValueError("Invalid choice. Please choose a number between 0 and 4.")

                self.trump = self.trump_opti[j - 1]
                break
            except ValueError as e:
                print(f"Input error: {e}. Try again.")

        print(f"\nStarting single-round game with trump suit: {self.trump if self.trump else 'No Trump'}")

        print(f"\nRound starts! Trump suit: {self.trump}")
        leading_player_index = 0
        total_tricks = len(self.players[0].hand)

        tricks_won = 0
        for round_num in range(total_tricks):
            print(f"\nTricks {round_num + 1}/{total_tricks}")
            leading_player_index = self.play_trick(leading_player_index)
            if self.players[leading_player_index] == self.highest_bidder:
                tricks_won += 1

            print("\nCurrent score:")
            for player in self.players:
                print(f"{player.name}: {player.tricks_won}/{total_tricks} tricks ")

        self.winning_team = max(self.team_scores, key=self.team_scores.get)
        self.winning_score = self.calculate_score(self.highest_bid, tricks_won)
        if not self.winning_score:
            print(f"\nTeam {self.winning_team + 1} loses the round with a score of {tricks_won}!")
            self.team_scores[self.winning_team] += self.winning_score
        else:
            print(f"\nTeam {self.winning_team + 1} wins the round with a score of {tricks_won}!")
            self.team_scores[self.winning_team] -= self.winning_score

    def play_game(self):

        self.deck = Deck()  # Initialize the deck
        self.deal_cards()
        self.__play_round()

        return [self.winning_team, self.winning_score]




if (__name__ == "__main__"):
    game = Whist(['a', 'b', 'c', 'd'])
    game.play_game()
