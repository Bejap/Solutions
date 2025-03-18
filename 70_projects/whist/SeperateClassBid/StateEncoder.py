import numpy as np
from Card import Card
from Player import Player


def create_state_vector(current_player: Player,
                        trick_cards: list[Card],
                        trump_suit: str,
                        highest_bid: int,
                        highest_bidder: Player,
                        lead_suit: str = None) -> np.ndarray:
    """
    Creates the state vector for the DQN input based on the current game state.
    """
    # Initialize all component vectors
    hand_vector = np.zeros(52)
    current_trick_vector = np.zeros(52)
    lead_suit_vector = np.zeros(4)
    trump_suit_vector = np.zeros(4)
    played_cards_vector = np.zeros(52)
    bid_vector = np.zeros(13)
    bidder_position_vector = np.zeros(4)

    # Convert player's hand to vector
    for card in current_player.hand:
        card_idx = card.suit_value * 13 + (card.rank_value - 2)
        hand_vector[card_idx] = 1

    # Convert current trick to vector
    if trick_cards:
        for card in trick_cards:
            card_idx = card.suit_value * 13 + (card.rank_value - 2)
            current_trick_vector[card_idx] = 1

    # Convert lead suit to one-hot vector
    if lead_suit:
        lead_suit_idx = Card.SUIT_VALUES[lead_suit]
        lead_suit_vector[lead_suit_idx] = 1

    # Convert trump suit to one-hot vector
    if trump_suit and trump_suit != "None":
        trump_suit_idx = Card.SUIT_VALUES[trump_suit]
        trump_suit_vector[trump_suit_idx] = 1

    # Convert seen cards to vector using your Card format
    for card_str in current_player.cards_seen:
        # Example card string format: "AoH" (Ace of Hearts)
        # Split by 'o' to get rank and suit
        parts = card_str.split('o')
        if len(parts) == 2:
            rank_str, suit_str = parts

            # Convert rank string to value
            rank_map = {
                '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
            }
            if rank_str in rank_map:
                rank_value = rank_map[rank_str] - 2

                # Convert suit string to value using your Card.SUIT_VALUES
                suit_map = {'H': 'Hearts', 'D': 'Diamonds', 'C': 'Clubs', 'S': 'Spades'}
                if suit_str in suit_map:
                    suit = suit_map[suit_str]
                    suit_value = Card.SUIT_VALUES[suit]

                    card_idx = suit_value * 13 + rank_value
                    played_cards_vector[card_idx] = 1

    # Encode bidding information
    if highest_bid >= 7:
        bid_vector[highest_bid - 7] = 1

    if highest_bidder:
        bidder_idx = (current_player.team * 2 + (0 if highest_bidder == current_player else 1)) % 4
        bidder_position_vector[bidder_idx] = 1

    # Concatenate all vectors
    state = np.concatenate([
        hand_vector,  # 52 dimensions
        current_trick_vector,  # 52 dimensions
        lead_suit_vector,  # 4 dimensions
        trump_suit_vector,  # 4 dimensions
        played_cards_vector,  # 52 dimensions
        bid_vector,  # 13 dimensions
        bidder_position_vector  # 4 dimensions
    ])

    return state


def print_state(state_vector: np.ndarray):
    """Prints the state vector in a human-readable format"""
    print("\n=== State Vector Analysis ===")

    # Print player's hand
    print("\nPlayer's Hand:")
    hand = state_vector[:52]
    for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
        suit_idx = Card.SUIT_VALUES[suit] * 13
        cards = []
        for i in range(13):
            if hand[suit_idx + i] == 1:
                rank_value = i + 2
                rank = next(k for k, v in Card.RANK_VALUES.items() if v == rank_value)
                cards.append(rank)
        if cards:
            print(f"{suit}: {', '.join(cards)}")

    # Print current trick
    print("\nCurrent Trick:")
    trick = state_vector[52:104]
    cards_in_trick = []
    for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
        suit_idx = Card.SUIT_VALUES[suit] * 13
        for i in range(13):
            if trick[suit_idx + i] == 1:
                rank_value = i + 2
                rank = next(k for k, v in Card.RANK_VALUES.items() if v == rank_value)
                cards_in_trick.append(f"{rank} of {suit}")
    print(', '.join(cards_in_trick) if cards_in_trick else "No cards played yet")

    # Print lead suit
    print("\nLead Suit:")
    lead_suit = state_vector[104:108]
    if any(lead_suit):
        suit_idx = np.argmax(lead_suit)
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        print(suits[suit_idx])
    else:
        print("No lead suit")

    # Print trump suit
    print("\nTrump Suit:")
    trump_suit = state_vector[108:112]
    if any(trump_suit):
        suit_idx = np.argmax(trump_suit)
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        print(suits[suit_idx])
    else:
        print("No trump")

    # Print played cards
    print("\nPreviously Played Cards:")
    played = state_vector[112:164]
    cards_played = []
    for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
        suit_idx = Card.SUIT_VALUES[suit] * 13
        for i in range(13):
            if played[suit_idx + i] == 1:
                rank_value = i + 2
                rank = next(k for k, v in Card.RANK_VALUES.items() if v == rank_value)
                cards_played.append(f"{rank} of {suit}")
    print(', '.join(cards_played) if cards_played else "No cards played yet")

    # Print bidding information
    print("\nBidding Information:")
    bid_vector = state_vector[164:177]
    if any(bid_vector):
        bid = np.argmax(bid_vector) + 7
        print(f"Highest bid: {bid}")
    else:
        print("No bids yet")

    bidder_position = state_vector[177:]
    if any(bidder_position):
        position = np.argmax(bidder_position)
        position_names = ["Current player", "Partner", "Left opponent", "Right opponent"]
        print(f"Highest bidder: {position_names[position]}")
    else:
        print("No highest bidder")
