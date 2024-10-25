import random
import time


class Card:
    '''
    A class representing a playing card.

    Each card has two attributes:
        suit: str - The suit of the card (Hearts, Diamonds, Clubs, Spades)
        rank: str - The rank of the card (2-10, J, Q, K, A)

    The card has no inherent value - the game rules determine
    how ranks are interpreted and compared.

    Example:
        card = Card("Hearts", "A")  # Creates the Ace of Hearts
    '''
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
    """
    Represents a standard 52-card deck with shuffling and dealing capabilities.

    Attributes:
        cards (list): List of Card objects representing remaining cards in deck
    """

    def __init__(self):
        self.cards: list[Card] = []
        suits: list[str] = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks: list[str] = ['2', '3', '4', '5', '6', '7', '8', '9', '10',
                 'Jack', 'Queen', 'King', 'Ace']
        # Create all 52 combinations of suits and ranks
        for suit in suits:
            for rank in ranks:
                self.cards.append(Card(suit, rank))

    def shuffle(self):
        """Randomize the order of cards in the deck."""
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        """
        Remove and return the specified number of cards from the deck.

        Args:
            num_cards (int): Number of cards to deal

        Returns:
            list: List of Card objects that were dealt
        """
        return [self.cards.pop() for _ in range(num_cards)]


class Player:
    def __init__(self, name: str, is_human: bool=False):
        self.name = name
        self.hand = []
        self.tricks_won = 0
        self.is_human = is_human
        self.cards_seen = set()  # Track cards played in the game

    def sort_hand(self):
        """Sort cards in hand by suit and rank."""
        self.hand.sort()

    def evaluate_card_strength(self, card: Card, trump_suit: str, lead_suit=None):
        """
        Evaluate the strength of a card based on game context.
        Returns a score from 0-100 where higher is better.
        """
        score: int = 0

        # Base strength from rank (0-50 points)
        score += (card.rank_value - 2) * 3.8  # Scales 2-14 to roughly 0-50

        # Trump suit bonus (20 points)
        if card.suit == trump_suit:
            score += 20

        # Leading suit bonus (15 points)
        if lead_suit and card.suit == lead_suit:
            score += 15

        # Adjust score based on cards seen
        remaining_higher_cards = 0
        for rank_value in range(card.rank_value + 1, 15):
            # Find the rank name for this value
            rank_name = next(k for k, v in Card.RANK_VALUES.items() if v == rank_value)
            potential_card = Card(card.suit, rank_name)
            if str(potential_card) not in self.cards_seen:
                remaining_higher_cards += 1

        # Reduce score if many higher cards remain
        score -= remaining_higher_cards * 2

        return max(0, min(score, 100))  # Ensure score is between 0-100

    def play_card(self, playable_cards: list[Card], trump_suit: str, lead_suit=None, played_cards=None):
        """Enhanced card playing logic for both human and AI players."""
        if self.is_human:
            # Sort playable cards for better display
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
                        self.hand.remove(card)
                        print(f"You played: {card.get_short_name()}")
                        time.sleep(1)
                        return card
                    else:
                        print("Invalid card number. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print(f"\n{self.name} is thinking...", end='', flush=True)
            time.sleep(0.5)

            # Smart AI logic
            best_card = None
            best_score = -1
            played_cards = played_cards or []

            for card in playable_cards:
                score = self.evaluate_card_strength(card, trump_suit, lead_suit)

                # Leading logic
                if not lead_suit:
                    # When leading, prefer non-trump high cards or low trumps
                    if card.suit != trump_suit:
                        score += card.rank_value * 2  # Prefer high cards when leading
                    else:
                        score -= card.rank_value  # Prefer low trumps when leading

                # Following logic
                else:
                    if card.suit == lead_suit:
                        # If we can follow suit, prefer winning cards
                        if played_cards:
                            highest_played = max((c for c in played_cards if c.suit == lead_suit),
                                                 key=lambda x: x.rank_value, default=None)
                            if highest_played and card.rank_value > highest_played.rank_value:
                                score += 15  # Bonus for winning the trick
                    elif card.suit == trump_suit:
                        # If we can't follow suit and have trump, prefer low trumps
                        score += 10  # Base bonus for playing trump
                        score -= card.rank_value  # Prefer lower trumps

                if score > best_score:
                    best_score = score
                    best_card = card

            self.hand.remove(best_card)
            print(f"\n{self.name} played: {best_card.get_short_name()}")
            time.sleep(1.5)
            return best_card


class Whist:
    def __init__(self, player_names: list[str], human_player_index: int=0):
        self.players = [
            Player(name, is_human=(i == human_player_index))
            for i, name in enumerate(player_names)
        ]
        self.deck = Deck()
        self.trump_suit = None

    def deal_cards(self):
        """Deal cards and sort hands."""
        self.deck.shuffle()
        cards_per_player = len(self.deck.cards) // len(self.players)
        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)
            player.sort_hand()  # Sort the hand after dealing
        self.trump_suit = random.choice(['Hearts', 'Diamonds', 'Clubs', 'Spades'])

    def play_trick(self, leading_player_index: int):
        trick_cards = []
        lead_suit = None
        winning_card = None
        winning_player = None

        print(f"\n{'=' * 50}")
        print(f"--- New Trick ---")
        print(f"Trump suit: {self.trump_suit}")
        time.sleep(1)

        for i in range(len(self.players)):
            current_player = self.players[(leading_player_index + i) % len(self.players)]
            print(f"\n{current_player.name}'s turn:")

            playable_cards = [card for card in current_player.hand if card.suit == lead_suit] if lead_suit else current_player.hand
            if not playable_cards:
                playable_cards = current_player.hand

            played_card = current_player.play_card(playable_cards, self.trump_suit, lead_suit, trick_cards)
            trick_cards.append(played_card)

            # Update cards seen for all players
            for player in self.players:
                player.cards_seen.add(str(played_card))

            if not lead_suit:
                lead_suit = played_card.suit
                print(f"Lead suit is now: {lead_suit}")
                time.sleep(0.5)

            if not winning_card:
                winning_card = played_card
                winning_player = current_player
            else:
                if (played_card.suit == winning_card.suit and played_card.rank_value > winning_card.rank_value) or \
                        (played_card.suit == self.trump_suit and winning_card.suit != self.trump_suit):
                    winning_card = played_card
                    winning_player = current_player

        winning_player.tricks_won += 1
        print(f"\n{winning_player.name} wins the trick with {winning_card.get_short_name()}!")
        time.sleep(1.5)
        return self.players.index(winning_player)

    def play_game(self):
        """Main game loop with enhanced scoring display."""
        self.deal_cards()
        print(f"\nGame starts! Trump suit: {self.trump_suit}")
        time.sleep(1)

        leading_player_index = 0
        total_tricks = len(self.players[0].hand)

        for round_num in range(total_tricks):
            print(f"\nRound {round_num + 1}/{total_tricks}")
            leading_player_index = self.play_trick(leading_player_index)

            print("\nCurrent scores:")
            for player in self.players:
                print(f"{player.name}: {player.tricks_won}/{round_num + 1} tricks " +
                      f"({(player.tricks_won / (round_num + 1) * 100):.1f}%)")
            time.sleep(1)

        winner = max(self.players, key=lambda x: x.tricks_won)
        return winner


# Example usage
if __name__ == "__main__":
    player_names = ["You", "Bob", "Charlie", "Diana"]
    game = Whist(player_names, human_player_index=0)
    winner = game.play_game()

    print("\nFinal Scores:")
    for player in game.players:
        print(f"{player.name}: {player.tricks_won} tricks")
    print(f"\nThe winner is {winner.name} with {winner.tricks_won} tricks!")