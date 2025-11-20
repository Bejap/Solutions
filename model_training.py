from whist import Whist
from simple_whist_DQN import DQNAgent
from ew_strategy import EWStrategy
from game_logger import GameLogger
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging

NUM_GAMES = 1000

epsilon = 1
EPSILON_DECAY = 0.996
MIN_EPSILON = 0.001
ARRAY_LENGTH = 52  # Full deck: 13 ranks * 4 suits
GAMMA_VALUES = [0.99, 0.95, 0.90, 0.85]
SAVE_EVERY = 500
LOG_GAME_EVERY = 50  # Save detailed game logs every 50 games
LOG_START_AFTER = 250  # Start logging after first 250 games

# Configure logging for monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # Initialize game logger
    game_logger = GameLogger(log_dir='game_logs')
    
    all_episode_rewards = []
    for episode in tqdm(range(1, NUM_GAMES + 1), ascii=True, unit='episodes'):
        trick_count = 0
        episode_rewards = [0, 0, 0, 0]

        start_state = game.reset()
        episode_rewards = [0, 0, 0, 0]
        done = False
        pending_transitions = []
        
        # Check if we should log this game (after first 250 games, every 50 games)
        should_log_game = (episode > LOG_START_AFTER and episode % LOG_GAME_EVERY == 0)
        
        # Start logging if this is a logged game
        if should_log_game:
            starting_player_idx = game.current_player_idx
            game_logger.start_game(episode, starting_player_idx, game.players, trump_suit='Spades')

        while trick_count < ARRAY_LENGTH and not done:  # Complete all tricks
            for _ in range(4):
                current_player_index = game.current_player_idx
                current_player = game.players[current_player_index]
                agent = agents[current_player_index]
                current_state = game.get_init_state()

                # Get valid actions based on follow suit rules
                valid_actions = game.get_valid_actions(current_player)
                
                # Calculate action_space as number of cards in hand (dynamic)
                action_space = len(current_player.hand)
                
                # Track decision type and certainty for logging
                decision_type = 'unknown'
                certainty = None

                # Only use agent for North (0) and South (2)
                if agent is not None:
                    a = np.random.random()
                    if a > epsilon:
                        # Agent making a decision based on Q-values
                        qs = agent.get_qs(current_state)
                        if valid_actions:
                            valid_q_values = [(card, qs[card]) for card in valid_actions if card < len(qs)]
                            if valid_q_values:
                                # Get the card index with highest Q-value
                                action, q_value = max(valid_q_values, key=lambda x: x[1])
                                decision_type = 'agent'
                                certainty = float(q_value)
                            else:
                                action = np.random.choice(valid_actions) if valid_actions else 0
                                decision_type = 'random'
                        else:
                            action = np.random.choice(valid_actions) if valid_actions else 0
                            decision_type = 'random'
                    else:
                        # Random exploration - but still must follow suit
                        action = np.random.choice(valid_actions) if valid_actions else 0
                        decision_type = 'random'
                else:
                    # Use strategic play for East (1) and West (3)
                    ew_strategy = ew_strategies[current_player_index]
                    action = ew_strategy.choose_action(current_player, valid_actions)
                    decision_type = 'strategy'

                # Peek at the card that will be played (don't remove it yet)
                if should_log_game and 0 <= action < len(current_player.hand):
                    # Sort hand to match what action() will return
                    current_player._sort_hand()
                    played_card = current_player.hand[action]
                    game_logger.log_card_played(current_player_index, played_card, decision_type, certainty)

                new_state, rewards, done = game.step(action)
                if rewards != 0:
                    episode_rewards[current_player_index] += rewards[current_player_index]

                # Only store transitions for agents
                if agent is not None and len(valid_actions) >= 1:
                    pending_transitions.append((current_state, action, None, new_state, False, current_player_index))

                if new_state is not None:
                    current_state = new_state

                if len(game.round_list) == 0:  # Trick is complete
                    trick_count += 1
                    
                    # Complete trick logging with winner if we're logging this game
                    if should_log_game and game.trick_winner is not None:
                        winner_idx = game.players.index(game.trick_winner)
                        game_logger.complete_trick(winner_idx)
                    
                    for s, a, _, ns, _, player_idx in pending_transitions:
                        if rewards != 0:
                            reward_value = rewards[player_idx]
                        else:
                            reward_value = 0

                        if sum(game.score_array) >= ARRAY_LENGTH:  # All tricks completed
                            done = True
                        
                        if agents[player_idx] is not None:
                            agents[player_idx].update_replay_memory((s, a, reward_value, ns, done))

                    for agent_idx, agent in enumerate(agents):
                        if agent is not None:
                            agent.train(done, trick_count)

                    pending_transitions = []

                if done:
                    break
        
        # End game logging if this was a logged game
        if should_log_game:
            game_logger.end_game(episode, game.score_array)
        
        all_episode_rewards.append(np.mean(episode_rewards))
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        for agent_idx, agent in enumerate(agents):
            if agent is not None:
                agent.train(True, trick_count)
        
        # Monitor reward statistics every 100 episodes
        if episode % 100 == 0:
            reward_stats = game.get_reward_stats()
            logger.info(f"Episode {episode} - Reward Stats: "
                       f"Agent 0 total: {reward_stats['agent_0_total']}, wins: {reward_stats['agent_0_wins']} | "
                       f"Agent 2 total: {reward_stats['agent_2_total']}, wins: {reward_stats['agent_2_wins']} | "
                       f"Tricks: {reward_stats['tricks_completed']}")

        if episode % SAVE_EVERY == 0:
            for i, agent in enumerate(agents):
                if agent is not None:
                    agent.save_agent(f"Weights/agent_player_{i}_ep{episode}.weights.h5")
                    agent.save_full_agent(f"Models/full_agent_player_{i}_ep{episode}.keras")
            logger.info(f"Models saved at episode {episode}")

    plt.plot(all_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("average reward")
    plt.title("learning over time")
    plt.grid(True)
    plt.show()
