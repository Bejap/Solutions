import Deck as dck
import Player as plr

class Whist:
    def __init__(self, player_names: list[str]):
        # Assign players to teams
        self.cards_played = None
        self.highest_bidder = None
        self.highest_bid = None
        self.active_players = None
        self.deck = None
        self.players = [plr.Player(name, is_human=False, team=i % 2) for i, name in enumerate(player_names)]
        self.count = 0
        self.trump_opti = ["Hearts", "Spades", "Diamonds", "Clubs", None]
        self.team_scores = {0: 0, 1: 0}
        self.winning_score = 0
        self.winning_team = None
        self.base_scores = {7: 1, 8: 3, 9: 6, 10: 10, 11: 15, 12: 22, 13: 30}
        self.file_name = "WhistData"

    def deal_cards(self):
        self.deck.shuffle()
        cards_per_player = 13  # CHANGE THIS
        self.count += 1

        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)
            player.sort_hand()

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
        if self.active_players == 1 and self.highest_bid == 6:
            for player in self.players:
                if not player.passed:
                    bid = player.make_bid(self.highest_bid)
                    if bid is not None and bid > self.highest_bid:
                        self.highest_bid = bid
                        self.highest_bidder = player

        if self.highest_bidder:
            print(f"Bidding ended. The winner is {self.highest_bidder.name} with a bid of {self.highest_bid}.")
            return self.highest_bidder
        else:
            print("No bids were placed. No winner.")
            return None

    def play_trick(self, leading_player_index: int):
        trick_cards = []
        lead_suit = None
        winning_card = None
        winning_player = None

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

        winning_player.tricks_won += 1
        self.team_scores[winning_player.team] += 1
        print(f"\n{winning_player.name} wins the trick for Team {winning_player.team + 1}!")

        return self.players.index(winning_player)

    def __play_round(self):
        s = self.start_bidding()
        if not s:
            print("No valid bids were placed. Skipping this round.")
            return  # Exit the method if no bidding winner

        print(s.name, "won, this is your hand", s.hand)
        while True:
            try:
                # Ask for user input
                j = int(input("Choose a trump suit: Hearts, Spades, Diamonds, Clubs, or None (1, 2, 3, 4, 0):\n"))
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
        self.highest_bidder = None
        self.highest_bid = None
        self.deck = dck.Deck()  # Initialize the deck
        self.deal_cards()
        self.reset_game()
        self.__play_round()


        return [self.winning_team, self.winning_score]

    def reset_game(self):
        """Resets the game state for a new match."""
        self.highest_bidder = None
        self.highest_bid = None
        for player in self.players:
            player.reset_for_new_game()


if __name__ == "__main__":
    game = Whist(['a', 'b', 'c', 'd'])
    game.play_game()
