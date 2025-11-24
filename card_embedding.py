"""
Card Embedding Module for Whist DQN

This module provides card embedding functionality that represents each card
as a learned vector embedding instead of one-hot encoding. This makes the
representation more flexible and independent of the total number of cards.

Key Benefits:
- Card representations are learned during training
- State size independent of number of cards dealt
- More compact and expressive representation
- Aggregated embeddings capture hand composition
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple
from constants import CARDS_PER_PLAYER


class CardEmbedding:
    """
    Card embedding layer that converts card IDs to learned vector representations.
    
    Each card is represented by:
    - Suit (0-3): Clubs, Diamonds, Hearts, Spades
    - Rank (0-12): 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
    
    The embedding combines suit and rank into a unified representation.
    """
    
    def __init__(self, embedding_dim: int = 8):
        """
        Initialize card embedding layer.
        
        Args:
            embedding_dim: Dimension of the embedding vector for each card (default: 8)
        """
        self.embedding_dim = embedding_dim
        
        # Separate embeddings for suit and rank
        # 4 suits + 1 for "no card/padding"
        self.suit_embedding = tf.keras.layers.Embedding(
            input_dim=5, 
            output_dim=embedding_dim // 2,
            name='suit_embedding'
        )
        
        # 13 ranks + 1 for "no card/padding"
        self.rank_embedding = tf.keras.layers.Embedding(
            input_dim=14,
            output_dim=embedding_dim // 2,
            name='rank_embedding'
        )
    
    def embed_card(self, card_id: int) -> tf.Tensor:
        """
        Convert a single card ID to its embedding.
        
        Args:
            card_id: Card identifier (0-51 for 52 cards, or 0-43 for 44 cards, etc.)
                    Use -1 or a value >= max_cards for padding/empty slots
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Decode card_id into suit and rank
        # For standard deck: card_id = suit * 13 + rank
        # suit: 0-3, rank: 0-12
        
        # Handle padding/empty cards
        if card_id < 0:
            suit_id = 4  # Padding suit
            rank_id = 13  # Padding rank
        else:
            suit_id = card_id // 13
            rank_id = card_id % 13
            
            # Clip to valid ranges
            suit_id = min(suit_id, 4)
            rank_id = min(rank_id, 13)
        
        # Get embeddings
        suit_emb = self.suit_embedding(suit_id)
        rank_emb = self.rank_embedding(rank_id)
        
        # Concatenate suit and rank embeddings
        card_emb = tf.concat([suit_emb, rank_emb], axis=-1)
        return card_emb
    
    def embed_card_list(self, card_ids: List[int], max_cards: int = None) -> tf.Tensor:
        """
        Embed a list of card IDs with optional padding.
        
        Args:
            card_ids: List of card identifiers
            max_cards: Maximum number of cards (for padding). If None, uses length of card_ids
        
        Returns:
            Tensor of shape (max_cards, embedding_dim) with embeddings
        """
        if max_cards is None:
            max_cards = len(card_ids)
        
        # Pad card_ids to max_cards length
        padded_ids = card_ids + [-1] * (max_cards - len(card_ids))
        padded_ids = padded_ids[:max_cards]  # Truncate if too long
        
        # Convert to tensor
        card_tensor = tf.constant(padded_ids, dtype=tf.int32)
        
        # Decode into suits and ranks
        suits = card_tensor // 13
        ranks = card_tensor % 13
        
        # Handle padding
        suits = tf.where(card_tensor < 0, 4, suits)  # 4 = padding
        ranks = tf.where(card_tensor < 0, 13, ranks)  # 13 = padding
        
        # Clip to valid ranges
        suits = tf.clip_by_value(suits, 0, 4)
        ranks = tf.clip_by_value(ranks, 0, 13)
        
        # Get embeddings
        suit_embs = self.suit_embedding(suits)
        rank_embs = self.rank_embedding(ranks)
        
        # Concatenate
        card_embs = tf.concat([suit_embs, rank_embs], axis=-1)
        return card_embs
    
    def aggregate_embeddings(self, card_embeddings: tf.Tensor, method: str = 'sum') -> tf.Tensor:
        """
        Aggregate card embeddings into a single vector.
        
        Args:
            card_embeddings: Tensor of shape (num_cards, embedding_dim)
            method: Aggregation method ('sum', 'mean', 'max', or 'attention')
        
        Returns:
            Aggregated embedding of shape (embedding_dim,)
        """
        if method == 'sum':
            return tf.reduce_sum(card_embeddings, axis=0)
        elif method == 'mean':
            return tf.reduce_mean(card_embeddings, axis=0)
        elif method == 'max':
            return tf.reduce_max(card_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


def cards_to_ids(cards: List) -> List[int]:
    """
    Convert card objects to card IDs.
    
    Args:
        cards: List of card objects with suit_value and rank_value attributes
    
    Returns:
        List of card IDs (integers)
    """
    card_ids = []
    for card in cards:
        # card_id = suit_value * 13 + (rank_value - 2)
        # Assuming rank_value ranges from 2-14, we subtract 2 to get 0-12
        card_id = card.suit_value * 13 + (card.rank_value - 2)
        card_ids.append(card_id)
    return card_ids


def create_embedded_state(
    hand_cards: List[int],
    round_cards: List[int],
    all_played_cards: List[int],
    player_tracking: List[List[int]],  # 4 players
    player_id: int,
    scores: List[float],
    embedding: CardEmbedding,
    max_hand_size: int = 13
) -> List[tf.Tensor]:
    """
    Create embedded state representation for the neural network.
    
    Args:
        hand_cards: List of card IDs in current player's hand
        round_cards: List of card IDs played in current round/trick
        all_played_cards: List of all card IDs played in the game so far
        player_tracking: List of 4 lists, each containing tracked cards for a player
        player_id: Current player ID (0-3)
        scores: List of 4 scores for each player
        embedding: CardEmbedding instance
        max_hand_size: Maximum hand size for padding
    
    Returns:
        List of tensors representing the embedded state
    """
    # Embed hand (aggregate to fixed size)
    hand_embs = embedding.embed_card_list(hand_cards, max_hand_size)
    hand_aggregated = embedding.aggregate_embeddings(hand_embs, method='sum')
    
    # Embed round cards (aggregate to fixed size)
    round_embs = embedding.embed_card_list(round_cards, max_cards=4)
    round_aggregated = embedding.aggregate_embeddings(round_embs, method='sum')
    
    # Embed all played cards (aggregate)
    played_embs = embedding.embed_card_list(all_played_cards)
    played_aggregated = embedding.aggregate_embeddings(played_embs, method='sum')
    
    # Player one-hot encoding
    player_encoding = tf.one_hot(player_id, depth=4)
    
    # Player card tracking (aggregate for each player)
    player_tracking_embs = []
    for player_cards in player_tracking:
        player_embs = embedding.embed_card_list(player_cards)
        player_agg = embedding.aggregate_embeddings(player_embs, method='sum')
        player_tracking_embs.append(player_agg)
    player_tracking_concat = tf.concat(player_tracking_embs, axis=-1)
    
    # Scores (as is)
    score_tensor = tf.constant(scores, dtype=tf.float32)
    
    # Return all components
    return [
        hand_aggregated,          # (embedding_dim,)
        round_aggregated,         # (embedding_dim,)
        played_aggregated,        # (embedding_dim,)
        player_encoding,          # (4,)
        player_tracking_concat,   # (embedding_dim * 4,)
        score_tensor              # (4,)
    ]


# Example usage and size calculation
if __name__ == "__main__":
    embedding_dim = 8
    
    print("Card Embedding System")
    print("=" * 60)
    print()
    print(f"Embedding dimension: {embedding_dim}")
    print()
    print("State Components:")
    print(f"  1. Hand aggregated:        {embedding_dim} dims")
    print(f"  2. Round aggregated:       {embedding_dim} dims")
    print(f"  3. Played aggregated:      {embedding_dim} dims")
    print(f"  4. Player encoding:        4 dims")
    print(f"  5. Player tracking (×4):   {embedding_dim * 4} dims")
    print(f"  6. Scores:                 4 dims")
    print()
    total_dims = embedding_dim * 3 + 4 + embedding_dim * 4 + 4
    print(f"Total state size: {total_dims} dims")
    print()
    print("Benefits:")
    print("  ✓ Independent of number of cards dealt")
    print("  ✓ Compact representation")
    print("  ✓ Learned card relationships")
    print("  ✓ Works with 9, 11, or 13 cards per player")
