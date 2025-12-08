"""
Modular Reward System Framework

This module provides a flexible framework for implementing and comparing
different reward system approaches for the Whist DQN training.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple


class BaseRewardSystem(ABC):
    """Abstract base class for reward systems."""
    
    def __init__(self, name: str):
        self.name = name
        self.episode_count = 0
    
    @abstractmethod
    def calculate_reward(self, game_state: Dict[str, Any], action: int, 
                        outcome: str, episode: int) -> float:
        """
        Calculate reward for a given state, action, and outcome.
        
        Args:
            game_state: Dictionary containing game state information
            action: Action taken
            outcome: Outcome string ('agent_wins', 'partner_wins', 'opponent_wins')
            episode: Current episode number
            
        Returns:
            Reward value
        """
        pass
    
    @abstractmethod
    def calculate_end_game_reward(self, team_won: bool, agent_tricks: int,
                                  partner_tricks: int, total_tricks: int) -> float:
        """
        Calculate end-game reward.
        
        Args:
            team_won: Whether the team won
            agent_tricks: Number of tricks won by agent
            partner_tricks: Number of tricks won by partner
            total_tricks: Total possible tricks
            
        Returns:
            End-game reward
        """
        pass
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_count += 1


class CurrentRewardSystem(BaseRewardSystem):
    """Current reward system implementation."""
    
    def __init__(self):
        super().__init__("current")
    
    def calculate_reward(self, game_state: Dict[str, Any], action: int,
                        outcome: str, episode: int) -> float:
        """Current system: Fixed trick rewards."""
        trick_rewards = {
            'agent_wins': +1.0,
            'partner_wins': +0.9,
            'opponent_wins': -1.1
        }
        return trick_rewards.get(outcome, 0.0)
    
    def calculate_end_game_reward(self, team_won: bool, agent_tricks: int,
                                  partner_tricks: int, total_tricks: int) -> float:
        """Current end-game formula."""
        tricks_won = agent_tricks
        max_tricks = total_tricks
        
        # Current formula
        base = (tricks_won - max_tricks) * (1 - tricks_won / max_tricks)
        team_bonus = +2.0 if team_won else 0.0
        
        return base + team_bonus


class HybridRewardSystem(BaseRewardSystem):
    """
    Hybrid Approach #11: Recommended custom reward system.
    
    Combines:
    - Adjusted trick rewards
    - Curriculum-based penalty scaling
    - Simplified end-game rewards
    - Win probability estimation (optional)
    """
    
    def __init__(self, curriculum_threshold: int = 3000, 
                 enable_win_prob: bool = False):
        super().__init__("hybrid_custom")
        self.curriculum_threshold = curriculum_threshold
        self.enable_win_prob = enable_win_prob
        self.win_prob_history = []
    
    def calculate_reward(self, game_state: Dict[str, Any], action: int,
                        outcome: str, episode: int) -> float:
        """Hybrid reward with curriculum scaling."""
        # 1. Base trick reward (adjusted magnitudes)
        trick_rewards = {
            'agent_wins': +1.0,
            'partner_wins': +0.6,  # Reduced from 0.8
            'opponent_wins': -0.8   # Reduced from -1.0
        }
        trick_reward = trick_rewards.get(outcome, 0.0)
        
        # 2. Scaled trump penalties (curriculum-based)
        trump_penalty = game_state.get('trump_penalty', 0.0)
        if episode < self.curriculum_threshold:
            penalty_scale = 1.0  # Full penalties early
        else:
            penalty_scale = 0.3  # Reduced penalties later
        
        scaled_penalty = trump_penalty * penalty_scale
        
        # 3. Win probability bonus (optional)
        win_bonus = 0.0
        if self.enable_win_prob:
            win_prob_change = self._estimate_win_prob_change(game_state)
            win_bonus = win_prob_change * 0.5
        
        # Combine components
        total_reward = trick_reward + scaled_penalty + win_bonus
        
        return total_reward
    
    def calculate_end_game_reward(self, team_won: bool, agent_tricks: int,
                                  partner_tricks: int, total_tricks: int) -> float:
        """Simplified end-game reward."""
        if team_won:
            return +5.0
        else:
            return -5.0
    
    def _estimate_win_prob_change(self, game_state: Dict[str, Any]) -> float:
        """
        Estimate change in win probability.
        
        Simple heuristic based on trick progress and team performance.
        """
        team_tricks = game_state.get('team_tricks', 0)
        total_played = game_state.get('total_tricks_played', 0)
        total_tricks = game_state.get('total_tricks', 13)
        
        if total_played == 0:
            return 0.0
        
        # Simple heuristic: are we ahead?
        tricks_remaining = total_tricks - total_played
        needed_to_win = (total_tricks // 2 + 1) - team_tricks
        
        if needed_to_win <= 0:
            # Already won
            current_prob = 1.0
        elif needed_to_win > tricks_remaining:
            # Can't win anymore
            current_prob = 0.0
        else:
            # Estimate based on progress
            current_prob = 1.0 - (needed_to_win / (tricks_remaining + 1))
        
        # Store and calculate change
        if len(self.win_prob_history) > 0:
            change = current_prob - self.win_prob_history[-1]
        else:
            change = 0.0
        
        self.win_prob_history.append(current_prob)
        
        return change
    
    def reset_episode(self):
        """Reset episode-specific tracking."""
        super().reset_episode()
        self.win_prob_history = []


class SparseRewardSystem(BaseRewardSystem):
    """Pure sparse rewards: only reward at game end."""
    
    def __init__(self):
        super().__init__("sparse")
    
    def calculate_reward(self, game_state: Dict[str, Any], action: int,
                        outcome: str, episode: int) -> float:
        """No intermediate rewards."""
        return 0.0
    
    def calculate_end_game_reward(self, team_won: bool, agent_tricks: int,
                                  partner_tricks: int, total_tricks: int) -> float:
        """Only game outcome matters."""
        return +1.0 if team_won else -1.0


class HierarchicalRewardSystem(BaseRewardSystem):
    """Hierarchical rewards with different scales for different levels."""
    
    def __init__(self):
        super().__init__("hierarchical")
    
    def calculate_reward(self, game_state: Dict[str, Any], action: int,
                        outcome: str, episode: int) -> float:
        """Multi-level rewards."""
        # Level 1: Card quality (small)
        card_rank_reward = game_state.get('card_rank', 0) * 0.1
        
        # Level 2: Trick outcome (medium)
        trick_rewards = {
            'agent_wins': +1.0,
            'partner_wins': +0.5,  # Less emphasis on partner
            'opponent_wins': -0.5   # Less severe penalty
        }
        trick_reward = trick_rewards.get(outcome, 0.0)
        
        return card_rank_reward + trick_reward
    
    def calculate_end_game_reward(self, team_won: bool, agent_tricks: int,
                                  partner_tricks: int, total_tricks: int) -> float:
        """Level 3: Game outcome (large, dominant signal)."""
        return +10.0 if team_won else -10.0


class CurriculumRewardSystem(BaseRewardSystem):
    """
    Curriculum-based reward scheduling.
    Transitions from dense shaped rewards to sparse rewards.
    """
    
    def __init__(self, curriculum_episodes: int = 3500):
        super().__init__("curriculum")
        self.curriculum_episodes = curriculum_episodes
    
    def calculate_reward(self, game_state: Dict[str, Any], action: int,
                        outcome: str, episode: int) -> float:
        """Blend shaped and true rewards based on progress."""
        # Calculate shaping weight (1.0 early, 0.0 late)
        alpha = max(0.0, 1.0 - episode / self.curriculum_episodes)
        
        # Dense shaped component
        shaped_rewards = {
            'agent_wins': +1.0,
            'partner_wins': +0.8,
            'opponent_wins': -1.0
        }
        shaped_reward = shaped_rewards.get(outcome, 0.0)
        
        # Sparse component (only wins/losses matter)
        sparse_reward = 0.0  # Accumulates at game end
        
        # Blend
        return alpha * shaped_reward + (1 - alpha) * sparse_reward
    
    def calculate_end_game_reward(self, team_won: bool, agent_tricks: int,
                                  partner_tricks: int, total_tricks: int) -> float:
        """Strong terminal signal."""
        # Calculate blending weight
        alpha = max(0.0, 1.0 - self.episode_count / self.curriculum_episodes)
        
        # Shaped end reward
        tricks_won = agent_tricks
        base_shaped = (tricks_won - total_tricks) * (1 - tricks_won / total_tricks)
        
        # Sparse end reward
        sparse = +10.0 if team_won else -10.0
        
        return alpha * base_shaped + (1 - alpha) * sparse


# Registry of available reward systems
REWARD_SYSTEMS = {
    'current': CurrentRewardSystem,
    'hybrid_custom': HybridRewardSystem,
    'sparse': SparseRewardSystem,
    'hierarchical': HierarchicalRewardSystem,
    'curriculum': CurriculumRewardSystem,
}


def get_reward_system(name: str, **kwargs) -> BaseRewardSystem:
    """
    Factory function to create reward system instances.
    
    Args:
        name: Name of the reward system
        **kwargs: Additional arguments for the reward system
        
    Returns:
        Instance of the requested reward system
    """
    if name not in REWARD_SYSTEMS:
        raise ValueError(f"Unknown reward system: {name}. Available: {list(REWARD_SYSTEMS.keys())}")
    
    return REWARD_SYSTEMS[name](**kwargs)
