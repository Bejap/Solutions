"""Game logger to save detailed game logs every N games."""

import os
from datetime import datetime


class GameLogger:
    """Logs detailed game information including hands and tricks."""
    
    POSITION_NAMES = {
        0: 'North',
        1: 'East', 
        2: 'South',
        3: 'West'
    }
    
    POSITION_SHORT = {
        0: 'N',
        1: 'E',
        2: 'S',
        3: 'W'
    }
    
    def __init__(self, log_dir='game_logs'):
        """Initialize the game logger.
        
        Args:
            log_dir: Directory to save game logs
        """
        self.log_dir = log_dir
        self.current_game_log = []
        self.starting_player_idx = None
        self.starting_hands = {}
        self.current_trick = []
        self.trick_number = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def start_game(self, episode_number, starting_player_idx, players, trump_suit='Spades', is_benchmark=False):
        """Start logging a new game.
        
        Args:
            episode_number: The episode/game number
            starting_player_idx: Index of the player who starts (0-3)
            players: List of player objects with hand attribute
            trump_suit: The trump suit for this game (default: 'Spades')
            is_benchmark: Whether this is a fixed-seed benchmark game
        """
        self.current_game_log = []
        self.starting_player_idx = starting_player_idx
        self.starting_hands = {}
        self.current_trick = []
        self.trick_number = 0
        self.trump_suit = trump_suit
        
        # Log header
        benchmark_marker = " [BENCHMARK - Fixed Seed]" if is_benchmark else ""
        self.current_game_log.append(f"=== Game {episode_number}{benchmark_marker} ===")
        self.current_game_log.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.current_game_log.append(f"Trump: {trump_suit}")
        self.current_game_log.append(f"Starting player: {self.POSITION_NAMES[starting_player_idx]}\n")
        
        # Log starting hands
        for i, player in enumerate(players):
            position_name = self.POSITION_NAMES[i]
            hand_str = self._format_hand(player.hand)
            self.starting_hands[i] = hand_str
            self.current_game_log.append(f"{position_name} hand: {hand_str}")
        
        self.current_game_log.append("")  # Blank line after hands
    
    def log_card_played(self, player_idx, card, decision_type='unknown', certainty=None):
        """Log a card played in the current trick.
        
        Args:
            player_idx: Index of the player (0-3)
            card: Card object that was played
            decision_type: Type of decision - 'agent', 'random', 'strategy'
            certainty: For agent decisions, the Q-value confidence (float)
        """
        self.current_trick.append((player_idx, card, decision_type, certainty))
    
    def complete_trick(self, winner_idx):
        """Complete and log the current trick with its winner.
        
        Args:
            winner_idx: Index of the player who won the trick (0-3)
        """
        if len(self.current_trick) == 4:
            self._log_trick(winner_idx)
    
    def _log_trick(self, winner_idx=None):
        """Log the completed trick.
        
        Args:
            winner_idx: Index of the player who won the trick (0-3)
        """
        self.trick_number += 1
        self.current_game_log.append(f"Trick {self.trick_number}:")
        
        # Log cards in the order they were played
        for player_idx, card, decision_type, certainty in self.current_trick:
            position_short = self.POSITION_SHORT[player_idx]
            
            # Format decision info
            if decision_type == 'agent' and certainty is not None:
                decision_info = f" [Agent: Q={certainty:.3f}]"
            elif decision_type == 'random':
                decision_info = " [Random exploration]"
            elif decision_type == 'strategy':
                decision_info = " [Strategy]"
            else:
                decision_info = ""
            
            self.current_game_log.append(f"  {position_short}: {card}{decision_info}")
        
        # Log the winner
        if winner_idx is not None:
            winner_name = self.POSITION_SHORT[winner_idx]
            self.current_game_log.append(f"\n{winner_name} winner")
        
        self.current_game_log.append("")  # Blank line after trick
        
        # Reset for next trick
        self.current_trick = []
    
    def end_game(self, episode_number, final_scores):
        """End the current game and save the log.
        
        Args:
            episode_number: The episode/game number
            final_scores: Array of final scores for each player
        """
        # Log final scores
        self.current_game_log.append("Final Scores:")
        for i, score in enumerate(final_scores):
            position_name = self.POSITION_NAMES[i]
            self.current_game_log.append(f"  {position_name}: {score} tricks")
        
        # Determine winner
        team_1_score = final_scores[0] + final_scores[2]  # North + South
        team_2_score = final_scores[1] + final_scores[3]  # East + West
        
        self.current_game_log.append("")
        if team_1_score > team_2_score:
            self.current_game_log.append(f"Winner: Team 1 (North-South) - {team_1_score} vs {team_2_score}")
        elif team_2_score > team_1_score:
            self.current_game_log.append(f"Winner: Team 2 (East-West) - {team_2_score} vs {team_1_score}")
        else:
            self.current_game_log.append(f"Tie: {team_1_score} vs {team_2_score}")
        
        # Save to file
        self._save_log(episode_number)
    
    def _save_log(self, episode_number):
        """Save the current game log to a file.
        
        Args:
            episode_number: The episode/game number
        """
        filename = f"game_{episode_number:04d}.txt"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.current_game_log))
        
        # Clear the log for next game
        self.current_game_log = []
    
    def _format_hand(self, hand):
        """Format a hand of cards as a string.
        
        Args:
            hand: List of Card objects
            
        Returns:
            String representation of the hand
        """
        if not hand:
            return "[]"
        
        # Sort by suit and rank
        sorted_hand = sorted(hand, key=lambda c: (c.suit_value, c.rank_value))
        
        # Group by suit
        suits = {}
        for card in sorted_hand:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card.rank)
        
        # Format as "Suit: rank1 rank2 rank3"
        parts = []
        for suit in ['Clubs', 'Diamonds', 'Hearts', 'Spades']:
            if suit in suits:
                ranks = ' '.join(suits[suit])
                parts.append(f"{suit}: {ranks}")
        
        return ' | '.join(parts)
