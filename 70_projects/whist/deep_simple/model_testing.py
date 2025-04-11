import numpy as np
from whist import Whist
from simple_whist_DQN import DQNAgent
import tensorflow as tf


def evaluate_agent_vs_randoms(agent: DQNAgent, num_games=100):
    model = tf.keras.models.load_model('Models/full_agent_player_3.keras')
    model.summary()
    # Opret en dummy-agent og erstat dens model med den indlæste model
    dummy_agent = DQNAgent(input_size=None)  # Du behøver ikke input_size her
    dummy_agent.model = model  # Udskift modellen
    dummy_agent.target_model = model  # Sørg for at target_model matcher også

    wins = 0
    total_tricks = 0
    agent_tricks = 0
    env = Whist([1, 2, 3, 4])
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
            print(valid_actions)
            if current_player_index == 0:
                qs = dummy_agent.get_qs(state)
                if valid_actions:
                    # Ensure that all card values are within the valid index range of qs
                    print("This is agent", current_player)
                    valid_q_values = [qs[card] for card in valid_actions if card < len(qs)]

                    if valid_q_values:
                        print("This is agent, q_values")
                        action = np.argmax(valid_q_values)
                        print(action)
                        # Get the corresponding card
                    else:
                        print("This is agent, random")
                        action = np.random.randint(action_space)
            else:
                print("This was a random action")
                action = np.random.randint(action_space)

            next_state, reward, done = env.step(action)


            max_reward = max(reward)
            for i in range(len(reward)):
                if reward[i] == max_reward:
                    trick_counts[i] += 1

            state = next_state
            print(done)
            count += 1

        agent_tricks += trick_counts[0]
        total_tricks += sum(trick_counts)
        if trick_counts[0] >= max(trick_counts):
            wins += 1

    print(f"\n🎯 Evalueret over {num_games} spil:")
    print(f" - Agenten vandt i alt {wins} ud af {total_tricks} stik")
    print(f" - Agenten var bedst i {wins} af spillene ({wins / num_games * 100:.1f}%)")

STATE_SIZE = 91
ACTION_SIZE = 13

evaluate_agent_vs_randoms(None, num_games=100)
