import gym
import numpy as np
import matplotlib.pyplot as plt

# Initialize the environment with render_mode to avoid warning
env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
STATS_EVERY = 100
SHOW_EVERY = 3000
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))


# Properly handle the tuple returned by reset()

for episode in range(EPISODES):
    initial_state, _ = env.reset()  # Unpack the tuple, ignoring the info dict
    discrete_state = get_discrete_state(initial_state)

    episode_reward = 0
    done = False
    truncated = False

    if episode % SHOW_EVERY == 0:  # Newer gym versions
        print(f"Episode: {episode}")
        # Only create render environment when needed
        render_env = gym.make("MountainCar-v0", render_mode="human")
        render_state, _ = render_env.reset()
        render_discrete_state = get_discrete_state(render_state)

        # Run a demonstration episode with the current policy
        render_done = False
        render_truncated = False
        while not render_done and not render_truncated:
            action = np.argmax(q_table[render_discrete_state])
            render_state, reward, render_done, render_truncated, _ = render_env.step(action)
            render_discrete_state = get_discrete_state(render_state)
        render_env.close()

    while not done and not truncated:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done and not truncated:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reason) - if goal position is achieved - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
    if episode % 10 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.grid(True)
plt.show()
