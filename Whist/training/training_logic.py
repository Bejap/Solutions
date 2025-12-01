from Whist.core.whist import Whist
from Whist.agents.simple_whist_DQN import DQNAgent
from Whist.agents.ew_strategy import EWStrategy
from Whist.logger.game_logger import GameLogger
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime
from Whist.utils.constants import (
    DEFAULT_NUM_GAMES,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    ARRAY_LENGTH,
    DEFAULT_GAMMA_VALUES,
    DEFAULT_SAVE_EVERY,
    NUM_PLAYERS,
    DQN_AGENT_POSITIONS,
    EAST,
    WEST,
    CARDS_PER_PLAYER,
    MODEL_SAVE_REWARD_THRESHOLD,
    ENABLE_PER_CARD_REWARD,
    MODEL_SAVE_CHECK_EVERY,
    MODEL_SAVE_MIN_GAMES,
    EXPLORATION_GAMES,
    EPSILON_DECAY_TYPE,
    EPSILON_STEP_DECAY_EPISODES,
    EPSILON_STEP_DECAY_VALUES
)
from Whist.agents.advanced_dqn import EpsilonScheduler


def _as_list(actions):
    """Return a list copy of actions if not None, otherwise None."""
    if actions is None:
        return None
    return list(actions)


def choose_agent_action(
    agent,
    current_state,
    epsilon: float,
    action_space: int,
    valid_actions,
    return_info: bool = False,
):
    """
    Choose an action for an agent using epsilon-greedy strategy.
    
    Args:
        agent: The DQN agent
        current_state: Current game state
        epsilon: Exploration rate
        action_space: Size of the action space
        valid_actions: List of valid action indices
        return_info: If True, return (action, is_exploration, q_value) tuple
    
    Returns:
        Chosen action index, or tuple (action, is_exploration, q_value) if return_info=True
    """
    if agent is None:
        raise ValueError("agent must not be None for choose_agent_action")
    if action_space <= 0:
        raise ValueError("action_space must be a positive integer")
    
    if np.random.random() > epsilon:
        # Exploit
        qs = agent.get_qs(current_state)
        best = _best_valid_action_from_qs(qs, valid_actions)
        if best is not None:
            if return_info:
                q_value = float(qs[best]) if best < len(qs) else None
                return best, False, q_value
            return best
        # No valid action in range -> fallback to uniform random
        action = _random_action(action_space, valid_actions=None)
        if return_info:
            return action, True, None
        return action
    else:
        # Explore: prefer sampling among valid_actions if present
        action = _random_action(action_space, valid_actions=valid_actions)
        if return_info:
            return action, True, None
        return action


def _random_action(action_space: int, valid_actions) -> int:
    """
    Sample a random action.

    If valid_actions is provided and non-empty, sample from it. Otherwise sample uniformly
    from [0, action_space).
    """
    if valid_actions:
        valid_list = _as_list(valid_actions)
        return int(np.random.choice(valid_list))
    # fallback to uniform sample over action_space
    return int(np.random.randint(action_space))


def _best_valid_action_from_qs(qs: np.ndarray, valid_actions):
    """
    Given Q-values array and an iterable of valid action indices, return the valid action index
    with the highest Q-value. If no valid actions or none in range, return None.
    """
    if not valid_actions:
        return None
    valid_list = [int(a) for a in valid_actions if 0 <= int(a) < len(qs)]
    if not valid_list:
        return None
    # Choose the action (original id) with max Q-value
    best_action = max(valid_list, key=lambda a: qs[a])
    return int(best_action)


class WhistTrainer:
    """Trainer class for Whist DQN agents."""
    
    def __init__(self, num_games=DEFAULT_NUM_GAMES, epsilon=DEFAULT_EPSILON, 
                 epsilon_decay=DEFAULT_EPSILON_DECAY, min_epsilon=DEFAULT_MIN_EPSILON, 
                 array_length=ARRAY_LENGTH, gamma_values=None, save_every=DEFAULT_SAVE_EVERY,
                 log_every=100, log_dir='game_logs', enable_per_card_reward=ENABLE_PER_CARD_REWARD,
                 model_save_threshold=MODEL_SAVE_REWARD_THRESHOLD, exploration_games=EXPLORATION_GAMES,
                 early_stopping_patience=100, early_stopping_min_delta=0.01,
                 epsilon_decay_type='exponential', use_dueling_dqn=False):
        """
        Initialize the Whist trainer.
        
        Args:
            num_games: Number of training episodes
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum exploration rate
            array_length: Number of cards (13 for Hearts 2-A)
            gamma_values: List of gamma values for different agents
            save_every: Save models every N episodes
            log_every: Log detailed game info every N episodes (default: 100)
            log_dir: Directory for game logs (default: 'game_logs')
            enable_per_card_reward: Enable per-card reward based on EW strategy (default: True)
            model_save_threshold: Only save model if average reward > this value
            exploration_games: Number of games with pure random exploration (default: 200)
            early_stopping_patience: Stop if no improvement for N episodes (0 = disabled)
            early_stopping_min_delta: Minimum change to qualify as improvement
            epsilon_decay_type: Type of epsilon decay ('exponential', 'linear', 'step', 'cosine')
            use_dueling_dqn: Use Dueling DQN architecture (default: False)
        """
        self.NUM_GAMES = num_games
        self.epsilon = epsilon
        self.EPSILON_DECAY = epsilon_decay
        self.MIN_EPSILON = min_epsilon
        self.ARRAY_LENGTH = array_length
        self.GAMMA_VALUES = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
        self.SAVE_EVERY = save_every
        self.LOG_EVERY = log_every
        self.enable_per_card_reward = enable_per_card_reward
        self.model_save_threshold = model_save_threshold
        self.exploration_games = exploration_games
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.epsilon_decay_type = epsilon_decay_type
        self.use_dueling_dqn = use_dueling_dqn
        
        # Early stopping tracking
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0
        
        # Initialize epsilon scheduler for advanced decay strategies
        self.epsilon_scheduler = EpsilonScheduler(
            initial_epsilon=epsilon,
            min_epsilon=min_epsilon,
            decay_type=epsilon_decay_type,
            decay_rate=epsilon_decay,
            total_episodes=num_games - exploration_games,
            step_episodes=EPSILON_STEP_DECAY_EPISODES,
            step_values=EPSILON_STEP_DECAY_VALUES
        )
        
        # Initialize game logger
        self.logger = GameLogger(log_dir=log_dir)
        
        # Initialize game and agents with per-card reward setting
        player_names = [1, 2, 3, 4]
        self.game = Whist(player_names, enable_per_card_reward=enable_per_card_reward)
        
        # Select agent type based on configuration
        if use_dueling_dqn:
            from Whist.agents.advanced_dqn import DuelingDQNAgent
            self.agents = [
                DuelingDQNAgent((self.ARRAY_LENGTH * 7) + 4 + 4, gamma=self.GAMMA_VALUES[i], agent_id=i) if i in DQN_AGENT_POSITIONS else None 
                for i in range(NUM_PLAYERS)
            ]
            print("Using Dueling DQN architecture with n-step returns and LR scheduling")
        else:
            # Only train agents for North (0) and South (2) positions, which are on the same team
            self.agents = [
                DQNAgent((self.ARRAY_LENGTH * 7) + 4 + 4, gamma=self.GAMMA_VALUES[i], agent_id=i) if i in DQN_AGENT_POSITIONS else None 
                for i in range(NUM_PLAYERS)
            ]
        
        # Create EW strategy players for positions 1 (East) and 3 (West)
        self.ew_strategies = {
            EAST: EWStrategy(2, self.game),  # Player 2 is at position 1 (East)
            WEST: EWStrategy(4, self.game)   # Player 4 is at position 3 (West)
        }
        
        # Set EW strategies on game for per-card reward calculation
        self.game.set_ew_strategies(self.ew_strategies)
        
        self.all_episode_rewards = []
    
    def train(self):
        """Run the training loop."""
        print(f"Starting training with DQN agents")
        print(f"Total episodes: {self.NUM_GAMES}")
        print(f"  - Exploration phase: {self.exploration_games} episodes (pure random)")
        print(f"  - Training phase: {self.NUM_GAMES - self.exploration_games} episodes (epsilon decay)")
        print(f"Logging will start after exploration phase (from episode {self.exploration_games + 1})")
        print(f"Model saving will start from episode {MODEL_SAVE_MIN_GAMES}")
        print()
        
        for episode in tqdm(range(1, self.NUM_GAMES + 1), ascii=True, unit='episodes'):
            trick_count = 0
            episode_rewards = [0, 0, 0, 0]
            
            # Determine if we're in exploration phase or training phase
            in_exploration = episode <= self.exploration_games
            
            # Only log games after exploration phase ends
            should_log = (not in_exploration) and (episode % self.LOG_EVERY == 0)

            start_state = self.game.reset()
            done = False
            pending_transitions = []
            
            # Start logging for this game if needed
            if should_log:
                self.logger.start_game(episode, self.game.current_player_idx, self.game.players, trump_suit='Spades')

            while trick_count < self.ARRAY_LENGTH and not done:  # Complete all tricks
                for _ in range(4):
                    current_player_index = self.game.current_player_idx
                    current_player = self.game.players[current_player_index]
                    agent = self.agents[current_player_index]
                    current_state = self.game.get_init_state()

                    # Get valid actions based on follow suit rules (returns indices 0 to len(hand)-1)
                    valid_actions = self.game.get_valid_actions(current_player)
                    
                    # Calculate action_space as number of cards in hand (dynamic)
                    action_space = len(current_player.hand)

                    # Only use agent for North (0) and South (2)
                    if agent is not None:
                        # During exploration phase, always use epsilon=1.0 (pure random)
                        current_epsilon = 1.0 if in_exploration else self.epsilon
                        
                        action, is_exploration, q_value = choose_agent_action(
                            agent, current_state, current_epsilon, action_space, valid_actions, return_info=True
                        )
                        # Set decision_type based on whether agent explored or exploited
                        if is_exploration:
                            decision_type = 'random'
                            certainty = None
                        else:
                            decision_type = 'agent'
                            certainty = q_value
                        
                        # Only calculate per-card reward during training phase (not during exploration)
                        if not in_exploration:
                            per_card_reward = self.game.calculate_per_card_reward(
                                current_player_index, action, valid_actions
                            )
                            episode_rewards[current_player_index] += per_card_reward
                        else:
                            per_card_reward = 0
                    else:
                        # Use strategic play for East (1) and West (3)
                        ew_strategy = self.ew_strategies[current_player_index]
                        action = ew_strategy.choose_action(current_player, valid_actions)
                        decision_type = 'strategy'
                        certainty = None
                        per_card_reward = 0
                    
                    # Get the card being played for logging
                    card_played = current_player.hand[action] if action < len(current_player.hand) else None

                    new_state, rewards, done = self.game.step(action)
                    
                    # Log the card played
                    if should_log and card_played is not None:
                        self.logger.log_card_played(current_player_index, card_played, decision_type=decision_type, certainty=certainty)
                    
                    if rewards != 0:
                        episode_rewards[current_player_index] += rewards[current_player_index]

                    # Only store transitions for agents - include per-card reward
                    if agent is not None and len(valid_actions) >= 1:
                        pending_transitions.append((current_state, action, per_card_reward, new_state, False, current_player_index))

                    if new_state is not None:
                        current_state = new_state

                    if len(self.game.round_list) == 0:  # Trick is complete
                        trick_count += 1
                        
                        # Log trick completion
                        if should_log:
                            winner_idx = self._get_last_trick_winner()
                            self.logger.complete_trick(winner_idx)
                        
                        for s, a, per_card_r, ns, _, player_idx in pending_transitions:
                            if rewards != 0:
                                # Combine trick reward with per-card reward
                                reward_value = rewards[player_idx] + per_card_r
                            else:
                                reward_value = per_card_r

                            if sum(self.game.score_array) >= self.ARRAY_LENGTH:  # All tricks completed
                                done = True
                            
                            if self.agents[player_idx] is not None:
                                self.agents[player_idx].update_replay_memory((s, a, reward_value, ns, done))

                        # Only train agents during training phase (not during exploration)
                        if not in_exploration:
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
            
            # Only decay epsilon after exploration phase
            if not in_exploration:
                # Use epsilon scheduler for advanced decay strategies
                training_episode = episode - self.exploration_games
                self.epsilon = self.epsilon_scheduler.get_epsilon(training_episode)

            # Only train agents at episode end during training phase (not during exploration)
            if not in_exploration:
                for agent_idx, agent_obj in enumerate(self.agents):
                    if agent_obj is not None:
                        agent_obj.train(True, trick_count)

            # Only save model if average reward is above threshold
            # Check every MODEL_SAVE_CHECK_EVERY games after MODEL_SAVE_MIN_GAMES
            if episode >= MODEL_SAVE_MIN_GAMES and episode % MODEL_SAVE_CHECK_EVERY == 0:
                # Calculate average reward over recent episodes
                recent_window = min(100, len(self.all_episode_rewards))
                avg_reward = np.mean(self.all_episode_rewards[-recent_window:])
                
                if avg_reward > self.model_save_threshold:
                    # Format avg_reward for filename (e.g., -3.45 -> "avgR-3.45")
                    avg_reward_str = f"avgR{avg_reward:.2f}"
                    # Generate timestamp for unique filenames
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    for i, agent_obj in enumerate(self.agents):
                        if agent_obj is not None:
                            agent_obj.save_agent(f"Weights/agent_player_{i}_ep{episode}_{avg_reward_str}_{timestamp}.weights.h5")
                            agent_obj.save_full_agent(f"Models/full_agent_player_{i}_ep{episode}_{avg_reward_str}_{timestamp}.keras")
                    print(f"\nEpisode {episode}: Saved models (avg reward: {avg_reward:.2f} > {self.model_save_threshold})")
                else:
                    print(f"\nEpisode {episode}: Skipped saving (avg reward: {avg_reward:.2f} <= {self.model_save_threshold})")
                
                # Early stopping check (only if enabled and after exploration phase)
                if self.early_stopping_patience > 0 and not in_exploration:
                    if avg_reward > self.best_avg_reward + self.early_stopping_min_delta:
                        # Improvement detected
                        self.best_avg_reward = avg_reward
                        self.episodes_without_improvement = 0
                        print(f"Episode {episode}: New best avg reward: {avg_reward:.2f}")
                    else:
                        # No improvement
                        self.episodes_without_improvement += MODEL_SAVE_CHECK_EVERY
                        print(f"Episode {episode}: No improvement ({self.episodes_without_improvement}/{self.early_stopping_patience} episodes)")
                        
                        if self.episodes_without_improvement >= self.early_stopping_patience:
                            print(f"\n{'='*60}")
                            print(f"Early stopping triggered after {episode} episodes")
                            print(f"Best average reward: {self.best_avg_reward:.2f}")
                            print(f"No improvement for {self.episodes_without_improvement} episodes")
                            print(f"{'='*60}\n")
                            break  # Exit training loop
    
    def _get_last_trick_winner(self):
        """Determine who won the last trick."""
        # Use the trick_winner attribute set by the game
        if self.game.trick_winner is not None:
            return self.game.players.index(self.game.trick_winner)
        return 0  # Default to first player if no winner set
    
    def plot_results(self, plot_dir='plots'):
        """Plot and save the training results.
        
        Args:
            plot_dir: Directory to save plots (default: 'plots')
        """
        import os
        from datetime import datetime
        
        # Create plots directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot 1: Average reward over episodes
        plt.figure(figsize=(12, 6))
        plt.plot(self.all_episode_rewards, alpha=0.7, label='Episode Reward')
        
        # Add rolling average
        if len(self.all_episode_rewards) >= 100:
            rolling_avg = np.convolve(self.all_episode_rewards, np.ones(100)/100, mode='valid')
            plt.plot(range(99, len(self.all_episode_rewards)), rolling_avg, 
                    color='red', linewidth=2, label='100-Episode Rolling Avg')
        
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("DQN Agent Learning Over Time")
        plt.legend()
        plt.grid(True)
        
        filename = os.path.join(plot_dir, f'reward_over_time_{timestamp}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
        # Plot 2: Reward distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.all_episode_rewards, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel("Average Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.axvline(np.mean(self.all_episode_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.all_episode_rewards):.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = os.path.join(plot_dir, f'reward_distribution_{timestamp}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
        # Plot 3: Cumulative reward
        plt.figure(figsize=(12, 6))
        cumulative_rewards = np.cumsum(self.all_episode_rewards)
        plt.plot(cumulative_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Over Training")
        plt.grid(True)
        
        filename = os.path.join(plot_dir, f'cumulative_reward_{timestamp}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
        
        print(f"\nAll plots saved to '{plot_dir}/' folder")
