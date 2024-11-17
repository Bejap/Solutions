import Card as card
import time

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

    def play_card(self, playable_cards: list[card.Card], trump_suit: str, lead_suit=None, played_cards=None):
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
                    if bid > current_highest_bid and bid <= 13:
                        self.bid = bid
                        return bid
                    else:
                        print(f"Bid must be higher than the current highest bid ({current_highest_bid}, but max 13).")
                except ValueError:
                    print("Please enter a valid number.")
            else:
                print("Invalid action. Please type 'bid' or 'pass'.")
