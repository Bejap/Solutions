"""
Whist Game Adapter for Card Embeddings

This module provides an adapter that wraps the regular Whist game
and converts states from one-hot encoding to embedded representations.
"""

import numpy as np
from whist import Whist
from card_embedding import CardEmbedding, cards_to_ids
from constants import CARDS_PER_PLAYER
import tensorflow as tf


class WhistEmbedded(Whist):
    """
    Whist game that provides embedded state representations.
    
    This class extends the regular Whist game and converts the one-hot
    encoded states into embedded representations suitable for the
    EmbeddedDQNAgent.
    """
    
    def __init__(self, player_names: list, embedding_dim: int = 8):
        """
        Initialize the embedded Whist game.
        
        Args:
            player_names: List of player names/IDs
            embedding_dim: Dimension of card embeddings (default: 8)
        """
        super().__init__(player_names)
        self.embedding_dim = embedding_dim
        self.card_embedding = CardEmbedding(embedding_dim=embedding_dim)
    
    def _convert_array_to_card_ids(self, card_array: list) -> list:
        """
        Convert a one-hot array representation to list of card IDs.
        
        Args:
            card_array: Array where non-zero indices represent cards
        
        Returns:
            List of card IDs (indices where array is non-zero)
        """
        card_ids = []
        for i, value in enumerate(card_array):
            if value != 0:
                card_ids.append(i)
        return card_ids
    
    def get_embedded_state(self):
        """
        Get the current game state as embedded representation.
        
        Returns:
            List of tensors: [hand_emb, round_emb, played_emb, player_enc, tracking_emb, scores]
        """
        # Get the regular state first
        regular_state = self.get_init_state()
        
        # Extract components from regular state
        # regular_state = [cards_array, round_array, hand_array, player_array, 
        #                  player1_cards, player2_cards, player3_cards, player4_cards, score_array]
        cards_array = regular_state[0]
        round_array = regular_state[1]
        hand_array = regular_state[2]
        player_array = regular_state[3]
        player1_cards = regular_state[4]
        player2_cards = regular_state[5]
        player3_cards = regular_state[6]
        player4_cards = regular_state[7]
        score_array = regular_state[8]
        
        # Convert to card IDs
        hand_card_ids = self._convert_array_to_card_ids(hand_array)
        round_card_ids = self._convert_array_to_card_ids(round_array)
        all_played_card_ids = self._convert_array_to_card_ids(cards_array)
        
        # Player tracking card IDs
        player_tracking_ids = [
            self._convert_array_to_card_ids(player1_cards),
            self._convert_array_to_card_ids(player2_cards),
            self._convert_array_to_card_ids(player3_cards),
            self._convert_array_to_card_ids(player4_cards)
        ]
        
        # Get current player ID
        current_player_id = np.argmax(player_array)
        
        # Create embeddings
        # Hand embedding (aggregated)
        if hand_card_ids:
            hand_embs = self.card_embedding.embed_card_list(hand_card_ids, max_cards=CARDS_PER_PLAYER)
            hand_aggregated = self.card_embedding.aggregate_embeddings(hand_embs, method='sum')
        else:
            hand_aggregated = tf.zeros(self.embedding_dim)
        
        # Round embedding (aggregated)
        if round_card_ids:
            round_embs = self.card_embedding.embed_card_list(round_card_ids, max_cards=4)
            round_aggregated = self.card_embedding.aggregate_embeddings(round_embs, method='sum')
        else:
            round_aggregated = tf.zeros(self.embedding_dim)
        
        # All played cards embedding (aggregated)
        if all_played_card_ids:
            played_embs = self.card_embedding.embed_card_list(all_played_card_ids)
            played_aggregated = self.card_embedding.aggregate_embeddings(played_embs, method='sum')
        else:
            played_aggregated = tf.zeros(self.embedding_dim)
        
        # Player one-hot encoding
        player_encoding = tf.one_hot(current_player_id, depth=4)
        
        # Player tracking embeddings (aggregate for each player)
        player_tracking_embs = []
        for player_card_ids in player_tracking_ids:
            if player_card_ids:
                player_embs = self.card_embedding.embed_card_list(player_card_ids)
                player_agg = self.card_embedding.aggregate_embeddings(player_embs, method='sum')
            else:
                player_agg = tf.zeros(self.embedding_dim)
            player_tracking_embs.append(player_agg)
        player_tracking_concat = tf.concat(player_tracking_embs, axis=-1)
        
        # Scores
        score_tensor = tf.constant(score_array, dtype=tf.float32)
        
        # Convert to numpy for consistency
        hand_aggregated = hand_aggregated.numpy()
        round_aggregated = round_aggregated.numpy()
        played_aggregated = played_aggregated.numpy()
        player_encoding = player_encoding.numpy()
        player_tracking_concat = player_tracking_concat.numpy()
        score_tensor = score_tensor.numpy()
        
        return [
            hand_aggregated,          # (embedding_dim,)
            round_aggregated,         # (embedding_dim,)
            played_aggregated,        # (embedding_dim,)
            player_encoding,          # (4,)
            player_tracking_concat,   # (embedding_dim * 4,)
            score_tensor              # (4,)
        ]
    
    def reset(self, seed=None):
        """
        Reset the game and return embedded state.
        
        Args:
            seed: Optional random seed
        
        Returns:
            Embedded state representation
        """
        # Call parent reset
        super().reset(seed)
        
        # Return embedded state
        return self.get_embedded_state()
    
    def step(self, action):
        """
        Execute a game step and return embedded state.
        
        Args:
            action: Action to take
        
        Returns:
            Tuple of (embedded_state, reward, done)
        """
        # Call parent step
        state, reward, done = super().step(action)
        
        # Convert state to embedded representation
        embedded_state = self.get_embedded_state()
        
        return embedded_state, reward, done
