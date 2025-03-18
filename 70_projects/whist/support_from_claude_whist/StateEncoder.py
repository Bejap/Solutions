import numpy as np



def create_state_vector(current_player: Player,
                        trick_cards: list[Card],
                        trump_suit: str,
                        highest_bid: int,
                        highest_bidder: Player,
                        lead_suit: str = None) -> np.ndarray:
    """
    Creates the state vector for the DQN input based on the current game state.

    Args:
        current_player: The player whose turn it is
        trick_cards: List of cards played in the current trick
        trump_suit: The current trump suit
        highest_bid: The current highest bid
        highest_bidder: The player who made the highest bid
        lead_suit: The leading suit for the current trick

    Returns:
        180-dimensional numpy array representing the game state
        (Previous 164 + 16 dimensions for bidding information)
    """
    # Initialize all component vectors
    hand_vector = np.zeros(52)  # Player's hand
    current_trick_vector = np.zeros(52)  # Current trick
    lead_suit_vector = np.zeros(4)  # Lead suit
    trump_suit_vector = np.zeros(4)  # Trump suit
    played_cards_vector = np.zeros(52)  # Previously played cards

    # New vectors for bidding information
    bid_vector = np.zeros(13)  # One-hot encoding for current highest bid (7-13)
    bidder_position_vector = np.zeros(4)  # One-hot encoding for highest bidder position

    # Convert player's hand to vector
    for card in current_player.hand:
        card_idx = card.suit_value * 13 + (card.rank_value - 2)
        hand_vector[card_idx] = 1

    # Convert current trick to vector
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

    # Convert seen cards to vector
    for card_str in current_player.cards_seen:
        # Parse the short name format (e.g., "AoH" for Ace of Hearts)
        rank_str = card_str[0]
        suit_str = card_str[-1]

        # Convert rank string to value
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                    '9': 9, '1': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        rank_value = rank_map[rank_str] - 2

        # Convert suit string to value
        suit_map = {'H': 0, 'D': 1, 'C': 2, 'S': 3}
        suit_value = suit_map[suit_str]

        card_idx = suit_value * 13 + rank_value
        played_cards_vector[card_idx] = 1

    # Encode bidding information
    if highest_bid >= 7:
        bid_vector[highest_bid - 7] = 1  # Subtract 7 because minimum bid is 7

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
        bid_vector,  # 13 dimensions (bids 7-13)
        bidder_position_vector  # 4 dimensions
    ])

    return state


# Add this method to your Whist class:
def get_state(self, current_player_index: int):
    """
    Gets the current state vector for the DQN.

    Args:
        current_player_index: Index of the current player

    Returns:
        numpy array representing the current game state
    """
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