import game2048
import numpy as np
from agent_2048 import DQNAgent
import tensorflow as tf

MODEL_PATH: str = ''
agent = DQNAgent(state_size=game2048.Game2048.BOARD_SIZE ** 2, action_size=4)
agent.model = tf.keras.models.load_model(MODEL_PATH)

scores = []
num_games = 100

for i in range(num_games):
    env = game2048.Game2048()
    current_state = env.get_state()
    done = False
    episode_score = 0

    while not done:
        action = np.argmax(agent.get_qs(current_state))
        new_state, reward, done = env.step(action)
        episode_score += reward
        current_state = new_state
        env.print_board()

    scores.append(episode_score)
    print(f"Game {i + 1} score: {episode_score}")

print(f"\nTested on {num_games} games.")
print(f"Average score: {np.mean(scores):.2f}")
print(f"Max score: {np.max(scores)}")
print(f"Min score: {np.min(scores)}")