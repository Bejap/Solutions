"""
Training Logic for Embedded DQN Agent

This module provides a trainer class for training the EmbeddedDQNAgent
on the Whist game using card embeddings instead of one-hot encoding.
"""

from whist_embedded import WhistEmbedded
from embedded_dqn_agent import EmbeddedDQNAgent
from ew_strategy import EWStrategy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from constants import (
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


def choose_embedded_agent_action(agent, current_state, epsilon: float, valid_actions) -> int:
    """
    Choose an action for an embedded agent using epsilon-greedy strategy.
    
    Args:
        agent: The embedded agent
        current_state: Current embedded state
        epsilon: Exploration rate
        valid_actions: List of valid action indices
    
    Returns:
        Chosen action index
    """
    if agent is None:
        raise ValueError("agent must not be None")
    
    return agent.choose_action(current_state, valid_actions, epsilon)


class EmbeddedWhistTrainer:
    """Trainer class for Embedded Whist DQN agents."""
    
    def __init__(self, embedding_dim=8, num_games=DEFAULT_NUM_GAMES, epsilon=DEFAULT_EPSILON, 
                 epsilon_decay=DEFAULT_EPSILON_DECAY, min_epsilon=DEFAULT_MIN_EPSILON, 
                 gamma_values=None, save_every=DEFAULT_SAVE_EVERY):
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
        """
        self.embedding_dim = embedding_dim
        self.NUM_GAMES = num_games
        self.epsilon = epsilon
        self.EPSILON_DECAY = epsilon_decay
        self.MIN_EPSILON = min_epsilon
        self.GAMMA_VALUES = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
        self.SAVE_EVERY = save_every
        
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
        print()
        
        for episode in tqdm(range(1, self.NUM_GAMES + 1), ascii=True, unit='episodes'):
            trick_count = 0
            episode_rewards = [0, 0, 0, 0]

            start_state = self.game.reset()
            done = False
            pending_transitions = []

            while trick_count < CARDS_PER_PLAYER and not done:  # Complete all tricks
                for _ in range(4):
                    current_player_index = self.game.current_player_idx
                    current_player = self.game.players[current_player_index]
                    agent = self.agents[current_player_index]
                    
                    # Get embedded state
                    current_state = self.game.get_embedded_state()

                    # Get valid actions based on actual hand
                    valid_actions = [i for i, value in enumerate(self.game.player_hand(current_player)) if value != 0]
                    
                    # Map valid actions to hand indices (0 to len(hand)-1)
                    hand_size = len(current_player.hand)
                    valid_hand_indices = list(range(hand_size))

                    # Only use agent for North (0) and South (2)
                    if agent is not None:
                        action = choose_embedded_agent_action(agent, current_state, self.epsilon, valid_hand_indices)
                    else:
                        # Use strategic play for East (1) and West (3)
                        ew_strategy = self.ew_strategies[current_player_index]
                        action = ew_strategy.choose_action(current_player, valid_hand_indices)

                    new_state, rewards, done = self.game.step(action)
                    if rewards != 0:
                        episode_rewards[current_player_index] += rewards[current_player_index]

                    # Only store transitions for agents
                    if agent is not None and len(valid_hand_indices) >= 1:
                        pending_transitions.append((current_state, action, None, new_state, False, current_player_index))

                    if new_state is not None:
                        current_state = new_state

                    if len(self.game.round_list) == 0:  # Trick is complete
                        trick_count += 1
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
