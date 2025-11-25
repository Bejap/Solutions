"""
Training Logic for Embedded DQN Agent

This module provides a trainer class for training the EmbeddedDQNAgent
on the Whist game using card embeddings instead of one-hot encoding.
"""

from Whist.core.whist_embedded import WhistEmbedded
from Whist.agents.embedded_dqn_agent import EmbeddedDQNAgent
from Whist.agents.ew_strategy import EWStrategy
from Whist.logger.game_logger import GameLogger
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from Whist.utils.constants import (
    DEFAULT_NUM_GAMES,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    CARDS_PER_PLAYER,
    DEFAULT_GAMMA_VALUES,
    DEFAULT_SAVE_EVERY,
    NUM_PLAYERS,
    DQN_AGENT_POSITIONS,
    EAST,
    WEST
)


def choose_embedded_agent_action(agent, current_state, epsilon: float, valid_actions, return_info: bool = False):
    """
    Choose an action for an embedded agent using epsilon-greedy strategy.
    
    Args:
        agent: The embedded agent
        current_state: Current embedded state
        epsilon: Exploration rate
        valid_actions: List of valid action indices
        return_info: If True, return (action, is_exploration, q_value) tuple
    
    Returns:
        Chosen action index, or tuple (action, is_exploration, q_value) if return_info=True
    """
    if agent is None:
        raise ValueError("agent must not be None")
    
    return agent.choose_action(current_state, valid_actions, epsilon, return_info=return_info)


class EmbeddedWhistTrainer:
    """Trainer class for Embedded Whist DQN agents."""
    
    def __init__(self, embedding_dim=8, num_games=DEFAULT_NUM_GAMES, epsilon=DEFAULT_EPSILON, 
                 epsilon_decay=DEFAULT_EPSILON_DECAY, min_epsilon=DEFAULT_MIN_EPSILON, 
                 gamma_values=None, save_every=DEFAULT_SAVE_EVERY, log_every=100, log_dir='game_logs_embedded'):
        """
        Initialize the Embedded Whist trainer.
        
        Args:
            embedding_dim: Dimension of card embeddings (default: 8)
            num_games: Number of training episodes
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum exploration rate
            gamma_values: List of gamma values for different agents
            save_every: Save models every N episodes
            log_every: Log detailed game info every N episodes (default: 100)
            log_dir: Directory for game logs (default: 'game_logs_embedded')
        """
        self.embedding_dim = embedding_dim
        self.NUM_GAMES = num_games
        self.epsilon = epsilon
        self.EPSILON_DECAY = epsilon_decay
        self.MIN_EPSILON = min_epsilon
        self.GAMMA_VALUES = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
        self.SAVE_EVERY = save_every
        self.LOG_EVERY = log_every
        
        # Initialize game logger
        self.logger = GameLogger(log_dir=log_dir)
        
        # Initialize embedded game
        player_names = [1, 2, 3, 4]
        self.game = WhistEmbedded(player_names, embedding_dim=embedding_dim)
        
        # Create embedded agents for North (0) and South (2) positions
        self.agents = [
            EmbeddedDQNAgent(embedding_dim=embedding_dim, gamma=self.GAMMA_VALUES[i], agent_id=i) 
            if i in DQN_AGENT_POSITIONS else None 
            for i in range(NUM_PLAYERS)
        ]
        
        # Create EW strategy players for positions 1 (East) and 3 (West)
        self.ew_strategies = {
            EAST: EWStrategy(2, self.game),  # Player 2 is at position 1 (East)
            WEST: EWStrategy(4, self.game)   # Player 4 is at position 3 (West)
        }
        
        self.all_episode_rewards = []
    
    def train(self):
        """Run the training loop with embedded representations."""
        print(f"Starting training with embedded agents (embedding_dim={self.embedding_dim})")
        print(f"State size: ~{self.embedding_dim * 7 + 8} dimensions (fixed, independent of cards)")
        print(f"Logging detailed games every {self.LOG_EVERY} episodes to '{self.logger.log_dir}/'")
        print()
        
        for episode in tqdm(range(1, self.NUM_GAMES + 1), ascii=True, unit='episodes'):
            trick_count = 0
            episode_rewards = [0, 0, 0, 0]
            should_log = (episode % self.LOG_EVERY == 0)

            start_state = self.game.reset()
            done = False
            pending_transitions = []
            
            # Start logging for this game if needed
            if should_log:
                self.logger.start_game(episode, self.game.current_player_idx, self.game.players, trump_suit='Spades')

            while trick_count < CARDS_PER_PLAYER and not done:  # Complete all tricks
                for _ in range(4):
                    current_player_index = self.game.current_player_idx
                    current_player = self.game.players[current_player_index]
                    agent = self.agents[current_player_index]
                    
                    # Get embedded state
                    current_state = self.game.get_embedded_state()

                    # Get valid actions based on follow suit rules (returns indices 0 to len(hand)-1)
                    valid_actions = self.game.get_valid_actions(current_player)

                    # Only use agent for North (0) and South (2)
                    if agent is not None:
                        action, is_exploration, q_value = choose_embedded_agent_action(
                            agent, current_state, self.epsilon, valid_actions, return_info=True
                        )
                        # Set decision_type based on whether agent explored or exploited
                        if is_exploration:
                            decision_type = 'random'
                            certainty = None
                        else:
                            decision_type = 'agent'
                            certainty = q_value
                    else:
                        # Use strategic play for East (1) and West (3)
                        ew_strategy = self.ew_strategies[current_player_index]
                        action = ew_strategy.choose_action(current_player, valid_actions)
                        decision_type = 'strategy'
                        certainty = None
                    
                    # Get the card being played for logging
                    card_played = current_player.hand[action] if action < len(current_player.hand) else None

                    new_state, rewards, done = self.game.step(action)
                    
                    # Log the card played
                    if should_log and card_played is not None:
                        self.logger.log_card_played(current_player_index, card_played, decision_type=decision_type, certainty=certainty)
                    
                    if rewards != 0:
                        episode_rewards[current_player_index] += rewards[current_player_index]

                    # Only store transitions for agents
                    if agent is not None and len(valid_actions) >= 1:
                        pending_transitions.append((current_state, action, None, new_state, False, current_player_index))

                    if new_state is not None:
                        current_state = new_state

                    if len(self.game.round_list) == 0:  # Trick is complete
                        trick_count += 1
                        
                        # Log trick completion
                        if should_log:
                            # Determine trick winner based on score changes
                            winner_idx = self._get_last_trick_winner()
                            self.logger.complete_trick(winner_idx)
                        
                        for s, a, _, ns, _, player_idx in pending_transitions:
                            if rewards != 0:
                                reward_value = rewards[player_idx]
                            else:
                                reward_value = 0

                            if sum(self.game.score_array) >= CARDS_PER_PLAYER:  # All tricks completed
                                done = True
                            
                            if self.agents[player_idx] is not None:
                                self.agents[player_idx].update_replay_memory((s, a, reward_value, ns, done))

                        for agent_idx, agent_obj in enumerate(self.agents):
                            if agent_obj is not None:
                                agent_obj.train(done, trick_count)

                        pending_transitions = []

                    if done:
                        break
            
            # End game logging
            if should_log:
                self.logger.end_game(episode, self.game.score_array)
                        
            self.all_episode_rewards.append(np.mean(episode_rewards))
            self.epsilon = max(self.MIN_EPSILON, self.epsilon * self.EPSILON_DECAY)

            for agent_idx, agent_obj in enumerate(self.agents):
                if agent_obj is not None:
                    agent_obj.train(True, trick_count)

            if episode % self.SAVE_EVERY == 0:
                for i, agent_obj in enumerate(self.agents):
                    if agent_obj is not None:
                        agent_obj.save_agent(f"Weights/embedded_agent_player_{i}_ep{episode}.weights.h5")
                        agent_obj.save_full_agent(f"Models/embedded_agent_player_{i}_ep{episode}.keras")
                print(f"\nEpisode {episode}: Saved embedded agent models")
    
    def _get_last_trick_winner(self):
        """Determine who won the last trick based on score changes."""
        # This is a simple approach - find the player with highest score change
        # In the actual game, the score_array shows cumulative wins
        scores = self.game.score_array
        max_score = max(scores)
        for i, score in enumerate(scores):
            if score == max_score:
                return i
        return 0  # Default to first player
    
    def plot_results(self):
        """Plot the training results."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Embedded Agent Learning Over Time")
        plt.grid(True)
        plt.savefig("embedded_training_results.png")
        print("Results saved to embedded_training_results.png")
        plt.show()
