import tensorflow as tf
import numpy as np
import whist as whist
from simple_whist_DQN import DQNAgent


def load_full_agent(filename):
    loaded_model = tf.keras.models.load_model(filename)
    print("Full agent model loaded successfully")

    agent = DQNAgent(input_size=loaded_model.input_shape[1])  # Create agent
    agent.model = loaded_model  # Assign loaded model to agent
    return agent


def load_agent_weights(agent, filename="whist_dqn_agent_.weights.h5"):
    agent.model.load_weights(filename)
    print("Agent weights loaded successfully")


def test_agent(agent, env, episodes=2):
    for episode in range(episodes):
        state = env.reset()  # Reset game for new episode
        if state is None:
            print("Error: State returned from env.reset() is None")
            continue  # Skip this episode if state is None

        done = False
        total_reward = 0
        # print(state)

        while not done:
            # Flatten and reshape the state
            state_input = agent._flat_the_state(state)
            state_input = np.array(state_input).reshape(1, -1)

            # Get Q-values
            q_values = agent.model.predict(state_input, verbose=0)[0]  # Get the first item from batch

            # Get current player
            current_player = env.players[env.count % 4]

            # Get valid actions
            valid_actions = [i for i, value in enumerate(env._player_hand(current_player)) if value == 1]

            if valid_actions:
                # Filter Q-values to only include valid actions
                valid_q_values = {action: q_values[action] for action in valid_actions if action < len(q_values)}

                if valid_q_values:
                    # Choose the valid action with highest Q-value
                    action = np.argmax(valid_q_values)
                else:
                    # Fallback to random valid action if no valid Q-values
                    action = np.random.choice(valid_actions)
            else:
                # No valid actions available
                print("Warning: No valid actions available")
                action = 0  # Default action

            # Take action in the game
            next_state, reward, done = env.step(action)

            total_reward += reward
            state = next_state
            print("next state", state)
            print(reward)

def test_agents(agents, env, episodes=2):
    for episode in range(episodes):
        state = env.reset()  # Reset game for new episode
        if state is None:
            print("Error: State returned from env.reset() is None")
            continue  # Skip this episode if state is None

        done = False
        total_rewards = [0, 0, 0, 0]  # Track total rewards for all agents
        step_rewards = []

        while not done:
            current_player_index = env.count % 4
            current_agent = agents[current_player_index]  # Select the correct agent
            current_player = env.players[current_player_index]

            # Flatten and reshape state for input
            state_input = np.array(current_agent._flat_the_state(state)).reshape(1, -1)

            # Get Q-values
            q_values = current_agent.model.predict(state_input, verbose=0)[0]  # Get the first item from batch

            # Get valid actions
            valid_actions = [i for i, value in enumerate(env._player_hand(current_player)) if value == 1]

            if valid_actions:
                # Filter Q-values to only include valid actions
                valid_q_values = {action: q_values[action] for action in valid_actions if action < len(q_values)}
                keys = list(valid_q_values.keys())

                if valid_q_values:
                    action_card = max(valid_q_values, key=valid_q_values.get)  # Pick action with highest Q-value
                    action = keys.index(action_card)
                else:
                    action = np.random.choice(valid_actions)  # Fallback to random valid action
            else:
                print("Warning: No valid actions available")
                action = 0  # Default action

            # Take action in the game
            next_state, reward, done = env.step(action)

            total_rewards[current_player_index] += reward
            step_rewards.append((current_player_index, action, reward))
            state = next_state  # Update state

            # print(f"Player {current_player_index} played action {action}, Reward: {reward}")
            print(state)

            # Display detailed reward breakdown
        print("Step-by-step rewards:")
        for step in step_rewards:
            player_idx, action_taken, reward_received = step
            print(f"    Player {player_idx}: Action {action_taken}, Reward {reward_received}")

        print(f"Episode {episode + 1} ended. Total rewards: {total_rewards}")




# player_names = ["1", "2", "3", "4"]
# env = whist.Whist(player_names)
# for i in range(4):
#     my_agent = load_full_agent(f"full_agent_player_{i}.keras")
# test_agent(my_agent, env)

# Load all four agents
agents = [load_full_agent(f"full_agent_player_{i}.keras") for i in range(4)]

# Initialize environment
player_names = ["1", "2", "3", "4"]
env = whist.Whist(player_names)

# Run the test function
test_agents(agents, env)
