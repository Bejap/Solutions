import Card as card
import time

class Player:

    def __init__(self, name: str, is_human: bool, team: int):
        self.name = name
        self.hand = []
        self.tricks_won = 0
        self.is_humna = is_human
        self.cards_seen = set()
        self.team = team

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