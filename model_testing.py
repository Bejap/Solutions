import numpy as np
from whist import Whist
from simple_whist_DQN import DQNAgent
from ew_strategy import EWStrategy
import tensorflow as tf
from constants import STATE_SIZE, ACTION_SIZE


def evaluate_agent_vs_randoms(agent: DQNAgent, num_games=100):
    model = tf.keras.models.load_model('Models/full_agent_player_3.keras')
    model.summary()
    dummy_agent = DQNAgent(input_size=None)
    dummy_agent.model = model
    dummy_agent.target_model = model

    wins = 0
    total_tricks = 0
    agent_tricks = 0
    env = Whist([1, 2, 3, 4])
    
    # Create EW strategy players
    ew_strategies = {
        1: EWStrategy(2, env),  # Player 2 is at position 1 (East)
        3: EWStrategy(4, env)   # Player 4 is at position 3 (West)
    }
    
    count = 0

    for _ in range(num_games):
        state = env.reset()
        done = False
        current_player_index = count % 4
        current_player = env.players[current_player_index]
        trick_counts = [0, 0, 0, 0]

        while not done:
            if 0 <= count < 4:
                action_space = 3
            if 0 <= count < 4:
                action_space = 2
            else:
                action_space = 1

            valid_actions = [i for i, value in enumerate(env.player_hand(current_player)) if value != 0]
            
            # Only use agent for North (position 0) and South (position 2)
            if current_player_index in [0, 2]:
                qs = dummy_agent.get_qs(state)
                if valid_actions:
                    valid_q_values = [qs[card] for card in valid_actions if card < len(qs)]
                    if valid_q_values:
                        action = np.argmax(valid_q_values)
                    else:
                        action = np.random.randint(action_space)
                else:
                    action = np.random.randint(action_space)
            else:
                # Use strategic play for East (1) and West (3)
                ew_strategy = ew_strategies[current_player_index]
                action = ew_strategy.choose_action(current_player, valid_actions)

            next_state, reward, done = env.step(action)

            max_reward = max(reward)
            for i in range(len(reward)):
                if reward[i] == max_reward:
                    trick_counts[i] += 1

            state = next_state
            count += 1

        agent_tricks += trick_counts[0]
        total_tricks += sum(trick_counts)
        if trick_counts[0] >= max(trick_counts):
            wins += 1

    print(f"\n🎯 Evaluated over {num_games} games:")
    print(f" - Agent won {wins} out of {total_tricks} tricks")
    print(f" - Agent was best in {wins} of the games ({wins / num_games * 100:.1f}%)")

evaluate_agent_vs_randoms(None, num_games=100)
