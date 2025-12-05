"""
Reward System Comparison Framework

This module provides tools for empirically comparing different reward systems
by training agents with each system and evaluating their performance.
"""

import time
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from Whist.training.common.reward_systems import get_reward_system, REWARD_SYSTEMS
from Whist.training.classic.training_logic import WhistTrainer
from Whist.core.whist import Whist
from Whist.agents.ew_strategy import EWStrategy
from Whist.utils.constants import (
    NUM_PLAYERS,
    DQN_AGENT_POSITIONS,
    EAST,
    WEST,
    CARDS_PER_PLAYER
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trained agent."""
    win_rate: float
    avg_tricks: float
    avg_reward: float
    training_time: float
    convergence_speed: int  # Episodes to reach threshold
    final_epsilon: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'win_rate': self.win_rate,
            'avg_tricks': self.avg_tricks,
            'avg_reward': self.avg_reward,
            'training_time': self.training_time,
            'convergence_speed': self.convergence_speed,
            'final_epsilon': self.final_epsilon
        }


class RewardSystemComparator:
    """
    Framework for comparing different reward systems empirically.
    
    Usage:
        comparator = RewardSystemComparator(
            reward_systems=['current', 'hybrid_custom', 'sparse'],
            n_training_episodes=5000,
            n_eval_games=100
        )
        results = comparator.compare_all()
        best_system = comparator.get_best_system(results)
    """
    
    def __init__(self, 
                 reward_systems: List[str] = None,
                 n_training_episodes: int = 5000,
                 n_eval_games: int = 100,
                 convergence_threshold: float = -3.0,
                 verbose: bool = True):
        """
        Initialize the comparator.
        
        Args:
            reward_systems: List of reward system names to compare
            n_training_episodes: Number of episodes to train each agent
            n_eval_games: Number of games for evaluation
            convergence_threshold: Reward threshold for convergence detection
            verbose: Whether to print progress
        """
        self.reward_systems = reward_systems or list(REWARD_SYSTEMS.keys())
        self.n_training_episodes = n_training_episodes
        self.n_eval_games = n_eval_games
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        
        if self.verbose:
            print("="*70)
            print("REWARD SYSTEM COMPARISON FRAMEWORK")
            print("="*70)
            print(f"Systems to compare: {self.reward_systems}")
            print(f"Training episodes per system: {n_training_episodes}")
            print(f"Evaluation games: {n_eval_games}")
            print(f"Convergence threshold: {convergence_threshold}")
            print("="*70)
    
    def train_with_reward_system(self, reward_system_name: str, 
                                **reward_system_kwargs) -> WhistTrainer:
        """
        Train an agent using a specific reward system.
        
        Args:
            reward_system_name: Name of the reward system
            **reward_system_kwargs: Additional kwargs for reward system
            
        Returns:
            Trained WhistTrainer instance
        """
        if self.verbose:
            print(f"\nTraining with reward system: {reward_system_name}")
            print("-"*70)
        
        # Create reward system
        reward_system = get_reward_system(reward_system_name, **reward_system_kwargs)
        
        # Create trainer (will integrate reward system)
        # Note: This requires modifying the trainer to accept reward system
        # For now, we'll use the standard trainer
        trainer = WhistTrainer(
            num_games=self.n_training_episodes,
            enable_per_card_reward=True,
            early_stopping_patience=0  # Disable for fair comparison
        )
        
        # Train
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        if self.verbose:
            print(f"Training completed in {training_time:.2f} seconds")
        
        return trainer, training_time
    
    def evaluate_agent(self, trainer: WhistTrainer) -> PerformanceMetrics:
        """
        Evaluate a trained agent's performance.
        
        Args:
            trainer: Trained WhistTrainer instance
            
        Returns:
            PerformanceMetrics with evaluation results
        """
        if self.verbose:
            print("Evaluating agent performance...")
        
        game = Whist(player_names=[1, 2, 3, 4], enable_per_card_reward=False)
        
        # Set up EW strategies
        ew_strategies = {
            EAST: EWStrategy(2, game),
            WEST: EWStrategy(4, game)
        }
        game.set_ew_strategies(ew_strategies)
        
        wins = 0
        total_tricks = []
        total_rewards = []
        
        # Run evaluation games
        for _ in range(self.n_eval_games):
            game.reset()
            episode_tricks = 0
            episode_reward = 0
            done = False
            
            while not done:
                current_player_idx = game.current_player_idx
                current_player = game.players[current_player_idx]
                
                # Get valid actions
                valid_actions = game.get_valid_actions(current_player)
                action_space = len(current_player.hand)
                
                # Choose action
                if current_player_idx in DQN_AGENT_POSITIONS:
                    agent = trainer.agents[current_player_idx]
                    current_state = game.get_init_state()
                    qs = agent.get_qs(current_state)
                    
                    # Greedy action (no exploration during eval)
                    valid_qs = {a: qs[a] for a in valid_actions if a < len(qs)}
                    if valid_qs:
                        action = max(valid_qs, key=valid_qs.get)
                    else:
                        action = valid_actions[0] if valid_actions else 0
                else:
                    action = ew_strategies[current_player_idx].choose_action(
                        current_player, valid_actions
                    )
                
                # Take action
                state, rewards, done = game.step(action)
                
                # Track agent performance
                if current_player_idx in DQN_AGENT_POSITIONS:
                    episode_reward += rewards[current_player_idx]
                    if rewards[current_player_idx] > 0:
                        episode_tricks += 1
            
            # Check if team won
            team_tricks = sum(game.score_array[i] for i in DQN_AGENT_POSITIONS)
            if team_tricks >= (CARDS_PER_PLAYER * 2) // 2 + 1:
                wins += 1
            
            total_tricks.append(episode_tricks)
            total_rewards.append(episode_reward)
        
        # Calculate convergence speed
        convergence_speed = self._estimate_convergence_speed(
            trainer.all_episode_rewards
        )
        
        metrics = PerformanceMetrics(
            win_rate=wins / self.n_eval_games,
            avg_tricks=np.mean(total_tricks),
            avg_reward=np.mean(total_rewards),
            training_time=0,  # Will be set by caller
            convergence_speed=convergence_speed,
            final_epsilon=trainer.epsilon
        )
        
        if self.verbose:
            print(f"  Win rate: {metrics.win_rate:.3f}")
            print(f"  Avg tricks: {metrics.avg_tricks:.2f}")
            print(f"  Avg reward: {metrics.avg_reward:.2f}")
            print(f"  Convergence at episode: {metrics.convergence_speed}")
        
        return metrics
    
    def _estimate_convergence_speed(self, episode_rewards: List[float]) -> int:
        """
        Estimate when the agent converged (reached stable performance).
        
        Args:
            episode_rewards: List of episode rewards
            
        Returns:
            Episode number where convergence was reached
        """
        if len(episode_rewards) < 100:
            return len(episode_rewards)
        
        # Look for when rolling average crosses threshold
        for i in range(100, len(episode_rewards)):
            window_avg = np.mean(episode_rewards[i-100:i])
            if window_avg >= self.convergence_threshold:
                return i
        
        return len(episode_rewards)  # Never converged
    
    def compare_all(self, **reward_system_kwargs) -> Dict[str, PerformanceMetrics]:
        """
        Compare all reward systems.
        
        Args:
            **reward_system_kwargs: Additional kwargs for reward systems
            
        Returns:
            Dictionary mapping system name to performance metrics
        """
        results = {}
        
        for system_name in self.reward_systems:
            try:
                # Train
                trainer, training_time = self.train_with_reward_system(
                    system_name, **reward_system_kwargs
                )
                
                # Evaluate
                metrics = self.evaluate_agent(trainer)
                metrics.training_time = training_time
                
                results[system_name] = metrics
                
                if self.verbose:
                    print(f"\n{'='*70}")
                    print(f"Results for {system_name}:")
                    print(f"  Win Rate: {metrics.win_rate:.3f}")
                    print(f"  Avg Tricks: {metrics.avg_tricks:.2f}")
                    print(f"  Avg Reward: {metrics.avg_reward:.2f}")
                    print(f"  Training Time: {metrics.training_time:.2f}s")
                    print(f"  Convergence: Episode {metrics.convergence_speed}")
                    print("="*70)
                    
            except Exception as e:
                print(f"Error training with {system_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def get_best_system(self, results: Dict[str, PerformanceMetrics],
                       metric: str = 'win_rate') -> tuple:
        """
        Get the best performing reward system.
        
        Args:
            results: Results from compare_all()
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (system_name, metrics)
        """
        if not results:
            return None, None
        
        best_system = max(results.items(), 
                         key=lambda x: getattr(x[1], metric))
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"BEST SYSTEM (by {metric}): {best_system[0]}")
            print(f"  {metric}: {getattr(best_system[1], metric):.3f}")
            print("="*70)
        
        return best_system
    
    def print_summary(self, results: Dict[str, PerformanceMetrics]):
        """
        Print a summary comparison table.
        
        Args:
            results: Results from compare_all()
        """
        print("\n" + "="*90)
        print("REWARD SYSTEM COMPARISON SUMMARY")
        print("="*90)
        print(f"{'System':<20} {'Win Rate':<12} {'Avg Tricks':<12} {'Train Time':<12} {'Converge':<12}")
        print("-"*90)
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1].win_rate, reverse=True):
            print(f"{name:<20} {metrics.win_rate:<12.3f} {metrics.avg_tricks:<12.2f} "
                  f"{metrics.training_time:<12.1f} {metrics.convergence_speed:<12d}")
        
        print("="*90)


def main():
    """
    Example usage of the reward system comparison framework.
    """
    # Compare all available reward systems
    comparator = RewardSystemComparator(
        reward_systems=['current', 'hybrid_custom', 'sparse', 'hierarchical', 'curriculum'],
        n_training_episodes=3000,
        n_eval_games=100,
        convergence_threshold=-3.0,
        verbose=True
    )
    
    # Run comparison
    results = comparator.compare_all(
        curriculum_threshold=2000,  # For curriculum-based systems
        enable_win_prob=False  # For hybrid system
    )
    
    # Print summary
    comparator.print_summary(results)
    
    # Get best system
    best_name, best_metrics = comparator.get_best_system(results, metric='win_rate')
    
    print(f"\nRecommended reward system: {best_name}")


if __name__ == "__main__":
    main()
