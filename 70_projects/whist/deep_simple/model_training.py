from whist import Whist
from simple_whist_DQN import DQNAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

EPISODES = 250

epsilon = 1
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
ARRAY_LENGTH = 13

if __name__ == "__main__":
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    # game.play()

    # for name, hand in game.get_player_hand().items():
    #     print(name, hand)
    agents = [DQNAgent((ARRAY_LENGTH * 7) + 4 + 4) for _ in range(4)]
    all_episode_rewards = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        print(f'Episode: {episode + 1}/{EPISODES}')
        print(epsilon)
        count = 0
        episode_rewards = [0, 0, 0, 0]

        current_state = game.reset()
        episode_rewards = [0, 0, 0, 0]
        done = False

        while count <= ARRAY_LENGTH - 1:
            pending_transitions = []
            for _ in range(4):
                current_player_index = count % 4
                current_player = game.players[current_player_index]
                agent = agents[current_player_index]  # Hent den rigtige agent

                if 0 <= count < 4:
                    action_space = 3
                if 0 <= count < 4:
                    action_space = 2
                else:
                    action_space = 1

                valid_actions = [i for i, value in enumerate(game.player_hand(current_player)) if value != 0]
                # print(valid_actions)

                if np.random.random() > epsilon:
                    qs = agent.get_qs(current_state)
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
                            action = np.random.randint(action_space)  # Fallback in case of an issue
                    else:
                        action = 0  # Default action
                else:
                    if valid_actions:
                        action = np.random.randint(action_space)  # Pick a random valid card
                    else:
                        action = 0  # Default action

                new_state, rewards, done = game.step(action)
                if rewards != 0:  # Check if rewards is not None
                    episode_rewards[current_player_index] += rewards[current_player_index]

                # Optionally train after each step

                if new_state is not None:
                    current_state = new_state  # Update state
                pending_transitions.append((current_state, action, None, new_state, False))
                # print(pending_transitions)

                if len(game.round_list) == 0:  # Trick is complete
                    for i, (s, a, _, ns, _) in enumerate(pending_transitions):
                        if rewards != 0:
                            reward_value = rewards[i]  # Get reward for this player
                            # Now add to replay memory with correct reward
                            if sum(game.score_array) == 3:
                                done = True
                            agents[i].update_replay_memory((s, a, reward_value, ns, done))
                            # print(f"Agent {i}: State: {s}, Action: {a}, Reward: {reward_value}, Next State: {ns}, Done: {done}")

                    for agent_idx, agent in enumerate(agents):
                        agent.train(done, count)

                    pending_transitions = []
                    # print(agent.model.input_shape)

                count += 1

                if done:
                    break
        all_episode_rewards.append(np.mean(episode_rewards))
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)  # Decay epsilon

        for agent in agents:
            agent.train(True, count)

    for i, agent in enumerate(agents):
        agent.save_agent(f"Weights/agent_player_{i}.weights.h5")
        agent.save_full_agent(f"Models/full_agent_player_{i}.keras")


    plt.plot(all_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Gennemsnitlig reward")
    plt.title("Læring over tid")
    plt.grid(True)
    plt.show()
