from Card import Card

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hand: list[Card] = []
        self.tricks_won: int = 0

    def play_card(self, playable_cards: list[Card], trump_suit: str, lead_suit: str=None, played_cards=None):

        if self.is_human:

            playable_cards.sort()
            while True:
                print(f"\nplayable cards: {playable_cards}")
                if lead_suit:
                    print(f'Lead Suit: {lead_suit}')

                try:
                    card_idx = int(input(f'Enter the number (1-{len(playable_cards)}) of card to play: ')) - 1
                    if 0 <= card_idx < len(playable_cards):
                        card = playable_cards[card_idx]
                        self.hand.remove(card)
                        print(f'You played: {card}')
                        return card
                    else:
                        print('Invalid card number')
                except ValueError:
                    print('Invalid card number')
        else:
            best_card = None
            best_score = -1
            played_cards = played_cards or []

            for card in playable_cards:
                score = self.evaluate_card_strength(card, trump_suit, lead_suit)

                if not lead_suit: # if lead_suit is None
                    if card.suit != trump_suit:
                        score += card.rank_value * 2
                    else:
                        score -= card.rank_value

