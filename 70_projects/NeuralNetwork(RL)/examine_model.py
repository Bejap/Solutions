from tensorflow.keras.models import load_model

# Load model
loaded_model = load_model('my_model_90.keras')

# View architecture
loaded_model.summary()

# View weights
for layer in loaded_model.layers:
    print(f"\n{layer.name} weights:")
    print(layer.get_weights())

# Add performance tracking
episode_rewards = []
episode_lengths = []

# for e in range(episodes):
#     state = env.reset()
#     state = np.reshape(state, [1, state_size])
#     total_reward = 0
#     steps = 0
#
#     for time in range(500):
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, state_size])
#         agent.remember(state, action, reward, next_state, done)
#         total_reward += reward
#         steps += 1
#         state = next_state
#
#         if done:
#             episode_rewards.append(total_reward)
#             episode_lengths.append(steps)
#             print(f"Episode {e + 1}/{episodes}")
#             print(f"Total Reward: {total_reward}")
#             print(f"Steps: {steps}")
#             print(f"Epsilon: {agent.epsilon:.2f}\n")
#             break
#
#     agent.replay(batch_size)
#
# # Visualize learning progress
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 5))
# plt.plot(episode_rewards)
# plt.title('Rewards per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.show()