import game2048
import numpy as np
from agent_2048 import DQNAgent
import tensorflow as tf

MODEL_PATH: str = 'models_3x3/2048__3562.70max_1629.74avg__445.70min__1747215225.keras'
agent = DQNAgent(state_size=game2048.Game2048.BOARD_SIZE, action_size=4)
agent.model = tf.keras.models.load_model(MODEL_PATH)

scores = []
num_games = 100
move_count = []

for i in range(num_games):
    env = game2048.Game2048()
    current_state = env.get_state()
    done = False
    episode_score = 0

    while not done:
        q_values = agent.get_qs(current_state)
        valid = env.get_valid_actions()
        masked = np.full_like(q_values, -np.inf)
        masked[valid] = q_values[valid]
        action = int(np.argmax(masked))
        new_state, reward, done = env.step(action)
        episode_score += reward
        current_state = new_state
        env.print_board()

    scores.append(episode_score)
    move_count.append(env.moves)
    print(f"Game {i + 1} score: {episode_score}")


print(f"\nTested on {num_games} games.")
print(f"Average score: {np.mean(scores):.2f}")
print(f"Average moves: {np.mean(move_count):.2f}")
print(f"Max score: {np.max(scores)}")
print(f"Max moves: {np.max(move_count)}")
print(f"Min score: {np.min(scores)}")
print(f"Min moves: {np.min(move_count)}")