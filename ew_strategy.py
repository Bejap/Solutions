"""
East-West Strategic Player Implementation

This module implements a rule-based strategy for East-West players following
bridge-like conventions. EW players use this strategy 80% of the time and play
randomly 20% of the time.
"""

import random
import whist_game as wg
from constants import EW_RANDOM_PLAY_PROBABILITY


class EWStrategy:
    """Strategic player for East-West positions following bridge-like rules."""
    
    def __init__(self, player_id, game_state):
        """
        Initialize the EW strategy.
        
        Args:
            player_id: The ID of the player (1-4)
            game_state: The current game state object
        """
        self.player_id = player_id
        self.game_state = game_state
        self.random_play_probability = EW_RANDOM_PLAY_PROBABILITY  # 20% random play
    
    def choose_action(self, player, valid_actions):
        """
        Choose an action based on strategic rules or random play.
        
        Args:
            player: The player object
            valid_actions: List of valid action indices (not used, we determine from hand)
            
        Returns:
            The chosen action index (0, 1, or 2 for position in sorted hand)
        """
        # Make sure hand is sorted
        player._sort_hand()
        
        if not player.hand:
            return 0
        
        # 20% of the time, play randomly
        if random.random() < self.random_play_probability:
            return random.randint(0, len(player.hand) - 1)
        
        # 80% of the time, use strategic play
        return self._strategic_action(player)
    
    def _strategic_action(self, player):
        """
        Choose an action based on strategic rules.
        
        Args:
            player: The player object (with sorted hand)
            
        Returns:
            The chosen action index (0-2, position in sorted hand)
        """
        # Get all cards in hand
        valid_cards = player.hand
        
        if not valid_cards:
            return 0
        
        # Determine if we're leading or following
        is_leading = len(self.game_state.round_list) == 0
        
        if is_leading:
            chosen_card = self._choose_lead(valid_cards)
        else:
            chosen_card = self._choose_follow(player, valid_cards)
        
        # Return the index of the chosen card in the sorted hand
        for i, card in enumerate(player.hand):
            if card == chosen_card:
                return i
        
        # Fallback to first card
        return 0
    
    def _choose_lead(self, valid_cards):
        """
        Choose a card to lead the trick.
        
        Rules:
        - Lead fourth-best from longest suit if >= 4 cards
        - Otherwise lead top of sequence (K from KQJ, Q from QJ)
        - Otherwise lead a singleton if aggressive play needed
        
        Args:
            valid_cards: List of valid card objects
            
        Returns:
            The chosen card object
        """
        # Group cards by suit
        suits = {}
        for card in valid_cards:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)
        
        # Sort each suit by rank
        for suit in suits:
            suits[suit].sort(key=lambda c: c.rank_value, reverse=True)
        
        # Find longest suit
        longest_suit = max(suits.keys(), key=lambda s: len(suits[s]))
        longest_suit_cards = suits[longest_suit]
        
        # If we have >= 4 cards in longest suit, lead fourth-best
        if len(longest_suit_cards) >= 4:
            # Fourth-best is the 4th highest card
            return longest_suit_cards[3]
        
        # Check for sequences and lead top of sequence
        for suit, cards in suits.items():
            if len(cards) >= 2:
                # Check if we have consecutive high cards
                for i in range(len(cards) - 1):
                    if cards[i].rank_value - cards[i + 1].rank_value == 1:
                        # Found a sequence, lead the top
                        return cards[i]
        
        # Lead from longest suit (top card)
        return longest_suit_cards[0]
    
    def _choose_follow(self, player, valid_cards):
        """
        Choose a card to follow in a trick.
        
        Rules:
        - Always follow suit if able (already enforced by valid_cards)
        - If can win: win only if it helps your side
        - If cannot win: play lowest card to conserve high cards
        
        Args:
            player: The player object
            valid_cards: List of valid card objects that can legally be played
            
        Returns:
            The chosen card object
        """
        # Get cards played so far in this trick
        trick_cards = self.game_state.round_list
        
        if not trick_cards:
            # Should not happen when following, but handle gracefully
            return valid_cards[0]
        
        # Get the led suit (first card in trick)
        led_suit = trick_cards[0][1].suit
        
        # Find the currently winning card
        winning_card = max(trick_cards, key=lambda t: t[1].rank_value)[1]
        
        # Separate our cards into those that can win and those that cannot
        winning_cards = [c for c in valid_cards if c.rank_value > winning_card.rank_value and c.suit == led_suit]
        losing_cards = [c for c in valid_cards if c not in winning_cards]
        
        # Check if our partner is currently winning
        partner_id = self._get_partner_id()
        partner_winning = any(t[0] == partner_id and t[1] == winning_card for t in trick_cards)
        
        if winning_cards:
            # We can win the trick
            if self._should_win_trick(player, winning_cards, partner_winning):
                # Win with the lowest card that can win (conserve high cards)
                return min(winning_cards, key=lambda c: c.rank_value)
            else:
                # Duck - play lowest card
                if losing_cards:
                    return min(losing_cards, key=lambda c: c.rank_value)
                else:
                    # We have no losing cards, play lowest winning card
                    return min(winning_cards, key=lambda c: c.rank_value)
        else:
            # We cannot win - play lowest card
            return min(valid_cards, key=lambda c: c.rank_value)
    
    def _should_win_trick(self, player, winning_cards, partner_winning):
        """
        Determine if we should win the trick.
        
        Heuristic: Win if you have the highest remaining card in the suit or
        if winning allows you to immediately lead a long suit you hold (>2 cards).
        Don't win if partner is already winning.
        
        Args:
            player: The player object
            winning_cards: List of cards that can win
            partner_winning: Boolean indicating if partner is currently winning
            
        Returns:
            Boolean indicating whether to try to win
        """
        # Don't win if partner is already winning
        if partner_winning:
            return False
        
        # Check if we have the highest remaining card in the led suit
        led_suit = self.game_state.round_list[0][1].suit
        
        # Get all cards we have in the led suit
        our_suit_cards = [c for c in player.hand if c.suit == led_suit]
        
        if our_suit_cards:
            highest_in_suit = max(our_suit_cards, key=lambda c: c.rank_value)
            # If our highest card is in winning_cards, we likely have the master card
            if highest_in_suit in winning_cards:
                return True
        
        # Check if we have a long suit (>2 cards) that we could lead
        suits = {}
        for card in player.hand:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)
        
        for suit, cards in suits.items():
            if len(cards) > 2:
                # We have a long suit, winning would give us the lead
                return True
        
        # Default: don't win unless necessary
        return False
    
    def _get_partner_id(self):
        """
        Get the partner's player ID.
        
        In Whist: 0-2 are partners, 1-3 are partners
        Player IDs are 1-4, so we need to map:
        - Player 1 (pos 0) partners with Player 3 (pos 2)
        - Player 2 (pos 1) partners with Player 4 (pos 3)
        
        Returns:
            The partner's player ID
        """
        # Convert to 0-indexed position
        position = self.player_id - 1
        
        # Partner is 2 positions away
        partner_position = (position + 2) % 4
        
        # Convert back to 1-indexed ID
        return partner_position + 1
