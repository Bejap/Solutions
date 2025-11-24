from whist import Whist
from simple_whist_DQN import DQNAgent
from ew_strategy import EWStrategy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from constants import (
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
    WEST
)


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
) -> int:
    if agent is None:
        raise ValueError("agent must not be None for choose_agent_action")
    if action_space <= 0:
        raise ValueError("action_space must be a positive integer")
    
    if np.random.random() > epsilon:
        # Exploit
        qs = agent.get_qs(current_state)
        best = _best_valid_action_from_qs(qs, valid_actions)
        if best is not None:
            return best
        # No valid action in range -> fallback to uniform random
        return _random_action(action_space, valid_actions=None)
    else:
        # Explore: prefer sampling among valid_actions if present
        return _random_action(action_space, valid_actions=valid_actions)


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
                 array_length=ARRAY_LENGTH, gamma_values=None, save_every=DEFAULT_SAVE_EVERY):
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
        """
        self.NUM_GAMES = num_games
        self.epsilon = epsilon
        self.EPSILON_DECAY = epsilon_decay
        self.MIN_EPSILON = min_epsilon
        self.ARRAY_LENGTH = array_length
        self.GAMMA_VALUES = gamma_values if gamma_values is not None else DEFAULT_GAMMA_VALUES
        self.SAVE_EVERY = save_every
        
        # Initialize game and agents
        player_names = [1, 2, 3, 4]
        self.game = Whist(player_names)
        
        # Only train agents for North (0) and South (2) positions, which are on the same team
        self.agents = [
            DQNAgent((self.ARRAY_LENGTH * 7) + 4 + 4, gamma=self.GAMMA_VALUES[i]) if i in DQN_AGENT_POSITIONS else None 
            for i in range(NUM_PLAYERS)
        ]
        
        # Create EW strategy players for positions 1 (East) and 3 (West)
        self.ew_strategies = {
            EAST: EWStrategy(2, self.game),  # Player 2 is at position 1 (East)
            WEST: EWStrategy(4, self.game)   # Player 4 is at position 3 (West)
        }
        
        self.all_episode_rewards = []
    
    def train(self):
        """Run the training loop."""
        for episode in tqdm(range(1, self.NUM_GAMES + 1), ascii=True, unit='episodes'):
            trick_count = 0
            episode_rewards = [0, 0, 0, 0]

            start_state = self.game.reset()
            done = False
            pending_transitions = []

            while trick_count < self.ARRAY_LENGTH and not done:  # Complete all tricks
                for _ in range(4):
                    current_player_index = self.game.current_player_idx
                    current_player = self.game.players[current_player_index]
                    agent = self.agents[current_player_index]
                    current_state = self.game.get_init_state()

                    # Get valid actions based on actual hand
                    valid_actions = [i for i, value in enumerate(self.game.player_hand(current_player)) if value != 0]
                    
                    # Calculate action_space as number of cards in hand (dynamic)
                    action_space = len(current_player.hand)

                    # Only use agent for North (0) and South (2)
                    if agent is not None:
                        action = choose_agent_action(agent, current_state, self.epsilon, action_space, valid_actions)
                    else:
                        # Use strategic play for East (1) and West (3)
                        ew_strategy = self.ew_strategies[current_player_index]
                        action = ew_strategy.choose_action(current_player, valid_actions)

                    new_state, rewards, done = self.game.step(action)
                    if rewards != 0:
                        episode_rewards[current_player_index] += rewards[current_player_index]

                    # Only store transitions for agents
                    if agent is not None and len(valid_actions) >= 1:
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

                            if sum(self.game.score_array) >= self.ARRAY_LENGTH:  # All tricks completed
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
                        agent_obj.save_agent(f"Weights/agent_player_{i}_ep{episode}.weights.h5")
                        agent_obj.save_full_agent(f"Models/full_agent_player_{i}_ep{episode}.keras")
    
    def plot_results(self):
        """Plot the training results."""
        plt.plot(self.all_episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("average reward")
        plt.title("learning over time")
        plt.grid(True)
        plt.show()
