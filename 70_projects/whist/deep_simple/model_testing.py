import numpy as np
from whist import Whist
from simple_whist_DQN import DQNAgent
import tensorflow as tf


def evaluate_agent_vs_randoms(agent: DQNAgent, num_games=100):
    model = tf.keras.models.load_model('Models/full_agent_player_0.keras')
    model.summary()
    # Create a dummy agent and replace its model with the loaded model
    dummy_agent = DQNAgent(input_size=None, gamma=None)
    dummy_agent.model = model
    dummy_agent.target_model = model

    wins = 0
    total_tricks = 0
    agent_tricks = 0
    env = Whist([1, 2, 3, 4])
    count = 0

    for game in range(num_games):
        state = env.reset()
        done = False
        trick_counts = [0, 0, 0, 0]

        while not done:
            if count < 4:
                action_space = 3
            elif 4 < count < 9:
                action_space = 2
            else:
                action_space = 1

            # Use env.count to determine the current player
            current_player_index = env.count % 4
            current_player = env.players[current_player_index]

            # Get valid actions for the current player
            valid_actions = [i for i, value in enumerate(env.player_hand(current_player)) if value != 0]
            print(f"Valid actions: {valid_actions}")

            # Determine action space based on the number of valid actions
            action_space = len(valid_actions) if valid_actions else 1

            if current_player_index == 0:  # Agent's turn
                qs = dummy_agent.get_qs(state)
                if valid_actions:
                    # Filter Q-values for valid actions only
                    valid_q_values = [qs[card] for card in valid_actions if card < len(qs)]

                    if valid_q_values:
                        action = np.argmax(valid_q_values)

                        print(f"Choosing action {action} (card index {valid_actions[action]})")
                    else:
                        print("Agent selecting randomly (no valid Q-values)")
                        action = np.random.choice(valid_actions)
                else:
                    print("No valid actions available!")
                    action = 0
            else:
                # Random player's turn
                print("Random player selecting action")
                action = np.random.randint(action_space)

            # Take action
            next_state, reward, done = env.step(action)
            count += 1

            # Check if a trick was completed (every 4 steps)
            if len(env.round_list) == 0 and env.count > 0:  # Round list was just emptied
                # The last trick winner is updated in the score_array
                for i in range(4):
                    if env.score_array[i] > trick_counts[i]:
                        trick_counts[i] += 1

            state = next_state
            print(f"Done: {done}")

        # After game is finished, get final trick counts from score_array
        agent_tricks += env.score_array[0]
        total_tricks += sum(env.score_array)

        # Check if agent won or tied for the win
        if env.score_array[0] >= max(env.score_array):
            wins += 1

    print(f"\n🎯 Evaluation over {num_games} games:")
    print(f" - Agent won {agent_tricks} out of {total_tricks} tricks")
    print(f" - Agent was best in {wins} of the games ({wins / num_games * 100:.1f}%)")


STATE_SIZE = 91
ACTION_SIZE = 13

evaluate_agent_vs_randoms(None, num_games=1000)