import re
from whist import Whist
from simple_whist_DQN import DQNAgent
from ew_strategy import EWStrategy
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

NUM_GAMES = 1000

epsilon = 1
EPSILON_DECAY = 0.996
MIN_EPSILON = 0.001
ARRAY_LENGTH = 13
GAMMA_VALUES = [0.99, 0.95, 0.90, 0.85]
SAVE_EVERY = 500

def select_action(
        agent,
        current_state,
        epsilon: float,
        action_space: int,
        valid_actions,
        current_player,
        current_player_index,
        ) -> int:
    if agent is not None:
        return choose_agent_action(agent, current_state, epsilon, action_space, valid_actions)

    else:
        # Use strategic play for East (1) and West (3)
        ew_strategy = ew_strategies[current_player_index]
        action = ew_strategy.choose_action(current_player, valid_actions)
        return action

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

if __name__ == "__main__":
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)

    # Only train agents for North (0) and South (2) positions, which are on the same team
    agents = [DQNAgent((ARRAY_LENGTH * 7) + 4 + 4, gamma=GAMMA_VALUES[i]) if i in [0, 2] else None for i in range(4)]
    
    # Create EW strategy players for positions 1 (East) and 3 (West)
    ew_strategies = {
        1: EWStrategy(2, game),  # Player 2 is at position 1 (East)
        3: EWStrategy(4, game)   # Player 4 is at position 3 (West)
    }
    
    all_episode_rewards = []
    for episode in tqdm(range(1, NUM_GAMES + 1), ascii=True, unit='episodes'):
        count = 0

        start_state = game.reset()
        episode_rewards = [0, 0, 0, 0]
        done = False
        pending_transitions = []

        while count != ARRAY_LENGTH - 1:
            for _ in range(4):
                current_player_index = game.current_player_idx
                current_player = game.players[current_player_index]
                agent = agents[current_player_index]
                current_state = game.get_init_state()

                action_space = 13 - count // 4

                valid_actions = [i for i, value in enumerate(game.player_hand(current_player)) if value != 0]

                action = select_action(agent, current_state, epsilon, action_space, valid_actions, current_player, current_player_index)

                new_state, rewards, done = game.step(action)
                if rewards != 0:
                    episode_rewards[current_player_index] += rewards[current_player_index]

                # Only store transitions for agents
                if agent is not None and len(valid_actions) >= 1:
                    pending_transitions.append((current_state, action, None, new_state, False, current_player_index))

                if new_state is not None:
                    current_state = new_state

                if len(game.round_list) == 0:  # Trick is complete
                    for s, a, _, ns, _, player_idx in pending_transitions:
                        if rewards != 0:
                            reward_value = rewards[player_idx]
                        else:
                            reward_value = 0

                        if sum(game.score_array) == 3:
                            done = True
                        
                        if agents[player_idx] is not None:
                            agents[player_idx].update_replay_memory((s, a, reward_value, ns, done))

                    for agent_idx, agent in enumerate(agents):
                        if agent is not None:
                            agent.train(done, count)

                    pending_transitions = []
                count += 1

                if done:
                    break
        all_episode_rewards.append(np.mean(episode_rewards))
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        for agent_idx, agent in enumerate(agents):
            if agent is not None:
                agent.train(True, count)

        if episode % SAVE_EVERY == 0:
            for i, agent in enumerate(agents):
                if agent is not None:
                    agent.save_agent(f"Weights/agent_player_{i}_ep{episode}.weights.h5")
                    agent.save_full_agent(f"Models/full_agent_player_{i}_ep{episode}.keras")

    plt.plot(all_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("average reward")
    plt.title("learning over time")
    plt.grid(True)
    plt.show()
