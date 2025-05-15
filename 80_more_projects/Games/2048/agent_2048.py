import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from game2048 import Game2048

DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 2500  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2048'
MIN_REWARD = 200 # For model save
MEMORY_FRACTION = 0.35

# Environment settings
EPISODES = 250*7

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.0005

#  Stats settings
AGGREGATE_STATS_EVERY = 25  # episodes


def process_state(state):
    return state.reshape(game.BOARD_SIZE, game.BOARD_SIZE, 1)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.input_shape = (state_size, state_size, 1)

        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0
        self.transition_count = 0

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse')
        return model

    def update_replay_memory(self, transition):
        if transition[2] > -25:
            self.replay_memory.append(transition)
        elif transition[2] == -150:
            self.replay_memory.append(transition)
        elif random.random() < 0.2:
            self.replay_memory.append(transition)
        if transition[4]:
            tran_copy = (transition[3], transition[1], -100, transition[3], True)
            self.replay_memory.append(tran_copy)



    def train(self, terminal_state):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = [transition[0] for transition in minibatch]
        current_states = np.array(current_states)
        current_qs_list = self.model.predict(current_states)

        new_current_states = [transition[3] for transition in minibatch]
        new_current_states = np.array(new_current_states)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (cur_s, act, rew, next_s, done_flag) in enumerate(minibatch):

            if not done_flag:
                max_future_q = np.max(future_qs_list[index])
                new_q = rew + DISCOUNT * max_future_q
            else:
                new_q = rew

            new_q = np.clip(new_q, -200, +200)

            current_qs = current_qs_list[index]
            current_qs[act] = new_q

            # And append to our training data
            X.append(cur_s)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        numpy_state = np.array(state)
        # state_reshaped = self.process_state(numpy_state)
        state_reshaped = np.expand_dims(numpy_state, axis=0)
        action_values = self.model.predict(state_reshaped, verbose=0)
        action_values = action_values[0]
        return action_values
        # return self.model.predict(np.array(state).reshape(1, -1), verbose=0)[0]


MODEL_PATH: str = 'models_3x3/2048__4941.70max_1364.14avg__123.70min__1747209497.keras'
agent = DQNAgent(state_size=Game2048.BOARD_SIZE ** 2, action_size=4)
# agent.model = tf.keras.models.load_model(MODEL_PATH)

ep_rewards = [-100]
reward = 0
best_avg_reward = -100

if __name__ == '__main__':
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes', desc="Training Progress"):
        print(f"Episode: {episode - 1}/{EPISODES}")
        print(epsilon)
        # time.sleep(1)

        agent_moves = 0
        episode_reward = 0
        step = 1

        game = Game2048()
        current_state = game.get_state()

        done = False
        while not done:
            if np.random.random() > epsilon:
                q_values = agent.get_qs(current_state)
                valid = game.get_valid_actions()
                masked = np.full_like(q_values, -np.inf)
                masked[valid] = q_values[valid]
                action = int(np.argmax(masked))
                agent_moves += 1
            else:
                masked_actions = game.get_valid_actions()
                action = np.random.choice(masked_actions)

            new_state, reward, done = game.step(action)
            reward /= 10
            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)
            current_state = new_state
            step += 1
            # print("\nThe agent accumulated this reward: ", reward)

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            print(average_reward)
            time.sleep(0.5)

            if average_reward >= MIN_REWARD and average_reward > best_avg_reward:
                best_avg_reward = average_reward  # Update the best seen
                agent.model.save(
                    f'models_4x4/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.keras'
                )

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        # print("\nMAX ", max(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        # print("MIN ", min(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        print(game.print_move_summary())
        print(game.print_board())
        print("The agent moved ", agent_moves, " times out of ", step, " moves.")

    plt.plot(ep_rewards[1:], label='Episode reward')  # [1:] to skip initial dummy -100
    rolling_mean = np.convolve(ep_rewards[1:], np.ones(10) / 10, mode='valid')
    plt.plot(range(9, len(ep_rewards) - 1), rolling_mean, label='Rolling avg (10)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training rewards over time')
    plt.grid(True)
    plt.show()
