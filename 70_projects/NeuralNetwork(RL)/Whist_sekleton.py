import numpy as np
from enum import Enum


class Suit(Enum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(Enum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def to_index(self):
        return self.suit.value * 13 + self.rank.value


def create_state_vector(
        hand: list[Card],
        current_trick: list[Card],
        lead_suit: Suit,
        trump_suit: Suit,
        played_cards: list[Card]
) -> np.ndarray:
    """
    Creates the state vector for the DQN input.

    Returns:
        164-dimensional numpy array representing the game state
    """
    # Initialize all component vectors
    hand_vector = np.zeros(52)
    current_trick_vector = np.zeros(52)
    lead_suit_vector = np.zeros(4)
    trump_suit_vector = np.zeros(4)
    played_cards_vector = np.zeros(52)

    # Encode hand
    for card in hand:
        hand_vector[card.to_index()] = 1

    # Encode current trick
    for card in current_trick:
        current_trick_vector[card.to_index()] = 1

    # Encode lead suit (one-hot)
    if lead_suit is not None:
        lead_suit_vector[lead_suit.value] = 1

    # Encode trump suit (one-hot)
    trump_suit_vector[trump_suit.value] = 1

    # Encode played cards
    for card in played_cards:
        played_cards_vector[card.to_index()] = 1

    # Concatenate all components
    state = np.concatenate([
        hand_vector,
        current_trick_vector,
        lead_suit_vector,
        trump_suit_vector,
        played_cards_vector
    ])

    return state


# Example usage
if __name__ == "__main__":
    # Example game state
    hand = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.CLUBS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.SEVEN),
        Card(Suit.CLUBS, Rank.EIGHT),
        Card(Suit.CLUBS, Rank.NINE),
        Card(Suit.CLUBS, Rank.TEN),
        Card(Suit.CLUBS, Rank.JACK),
        Card(Suit.CLUBS, Rank.ACE),
    ]

    current_trick = [
        Card(Suit.HEARTS, Rank.SEVEN)
    ]

    lead_suit = Suit.HEARTS
    trump_suit = Suit.SPADES

    played_cards = [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.HEARTS, Rank.THREE)
    ]

    state = create_state_vector(
        hand,
        current_trick,
        lead_suit,
        trump_suit,
        played_cards
    )

    print(f"State vector shape: {state.shape}")  # Should be (164,)

    # Print some meaningful sections of the state
    print("\nHand cards (first 52 elements):")
    print(state[:52].reshape(4, 13))  # Reshape to see suits × ranks

    print("\nLead suit one-hot encoding:")
    print(state[104:108])  # Elements 104-107 represent lead suit

    print("\nTrump suit one-hot encoding:")
    print(state[108:112])  # Elements 108-111 represent trump suit