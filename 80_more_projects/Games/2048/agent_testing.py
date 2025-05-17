import game2048
import numpy as np
from agent_2048 import DQNAgent
import tensorflow as tf
import csv
import matplotlib.pyplot as plt

MODEL_PATH: str = 'models_4x4/2048__1896.36max__624.35avg___44.46min__1747474811.keras'
agent = DQNAgent(state_size=game2048.Game2048.BOARD_SIZE, action_size=4)
agent.model = tf.keras.models.load_model(MODEL_PATH)

scores = []
num_games = 50
move_count = []
move_counter_summaries = []
avg_move = []

for i in range(num_games):
    env = game2048.Game2048()
    current_state = env.get_state()
    done = False
    episode_score = 0
    step = 0

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
        step += 1

    scores.append(episode_score)
    move_count.append(env.moves)
    move_counter_summaries.append(env.move_counter.copy())
    print(f"Game {i + 1} score: {episode_score}")

    up_count = env.move_counter['up']
    up_avg = up_count / step
    down_count = env.move_counter['down']
    down_avg = down_count / step
    left_count = env.move_counter['left']
    left_avg = left_count / step
    right_count = env.move_counter['right']
    right_avg = right_count / step

    ratio = abs(up_avg - down_avg) + abs(left_avg - right_avg)

    avg_move.append(ratio)

with open('eval_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Game', 'Score', 'Moves', 'up', 'down', 'left', 'right', 'Sum']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(num_games):
        row = {
            'Game': i + 1,
            'Score': scores[i],
            'Moves': move_count[i],
            **move_counter_summaries[i]
        }
        writer.writerow(row)

print(f"\nTested on {num_games} games.")
print(f"Average score: {np.mean(scores):.2f}")
print(f"Average moves: {np.mean(move_count):.2f}")
print(f"Max score: {np.max(scores)}")
print(f"Max moves: {np.max(move_count)}")
print(f"Min score: {np.min(scores)}")
print(f"Min moves: {np.min(move_count)}")

plt.figure(figsize=(10, 5))
rolling_move_avg = np.convolve(avg_move[1:], np.ones(10) / 10, mode='valid')
plt.plot(range(9, len(avg_move) - 1), rolling_move_avg, label='Rolling avg move (10)', color='orange')
plt.xlabel('Episode')
plt.ylabel('Average Moves')
plt.legend()
plt.title('Average Moves Over Time')
plt.grid(True)
plt.show()
