import gym
import tensorflow as tf
import numpy as np

GAMMA = 0.99
LAMBDA = 0.95
CLIP_RATIO = 0.2
POLICY_LR = 3e-4
VALUE_LR = 1e-3
TRAIN_EPOCHS = 10
STEPS_PER_EPOCH = 2048
MINI_BATCH_SIZE = 64


env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Actor: outputs probabilities over actions
actor = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(obs_dim,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(n_actions, activation='softmax')
])

# Critic: outputs state value
critic = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(obs_dim,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1)
])

actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


def sample_action(obs):
    logits = actor(obs[np.newaxis])  # (1, n_actions)
    log_probs = tf.math.log(logits)
    action = tf.random.categorical(log_probs, 1)
    action_idx = tf.squeeze(action, axis=1)  # shape: (1,) → (scalar)
    log_prob = tf.reduce_sum(log_probs[0] * tf.one_hot(action_idx, n_actions))
    return int(action_idx.numpy()), log_prob

def compute_gae(rewards, values, dones, next_value):
    advs = []
    gae = 0
    values = np.append(values, next_value)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advs.insert(0, gae)
    return np.array(advs)

def ppo_loss(old_log_probs, new_log_probs, advantages):
    ratio = tf.exp(old_log_probs - new_log_probs)
    clipped = tf.clip_by_value(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)
    return -tf.reduce_mean(tf.minimum(ratio * advantages, clipped) * advantages)

def train():
    obs_buf, act_buf, adv_buf, ret_buf, logp_buf = [], [], [], [], []

    obs = env.reset()[0]
    done = False
    ep_len = 0
    ep_ret = 0
    values = []
    rewards = []
    dones = []
    log_probs = []

    for step in range(STEPS_PER_EPOCH):
        obs_tensor = tf.convert_to_tensor(obs_buf, dtype=tf.float32)
        action, log_prob = sample_action(obs)
        value = critic(obs_tensor[None])[0, 0].numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(action)
        rewards.append(reward)
        values.append(value)
        dones.append(done)
        log_probs.append(log_prob)

        ep_len += 1
        ep_ret += reward

        obs = next_obs

        if done or step == STEPS_PER_EPOCH - 1:
            next_value = 0 if done else critic(tf.convert_to_tensor(obs[None], dtype=tf.float32))[0, 0].numpy()
            advantages = compute_gae(rewards, values, dones, next_value)
            returns = advantages + np.array(values)

            adv_buf.extend(advantages)
            ret_buf.extend(returns)
            logp_buf.extend([lp.numpy() for lp in log_probs])

            obs, _ = env.reset()[0]
            rewards, values, dones, log_probs = [], [], [], []
            ep_len = 0
            ep_ret = 0

    adv_buf = np.array(adv_buf)
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

    dataset = tf.data.Dataset.from_tensor_slices((
        np.array(obs_buf, dtype=np.float32),
        np.array(act_buf, dtype=np.int32),
        np.array(logp_buf, dtype=np.float32),
        np.array(adv_buf, dtype=np.float32),
        np.array(ret_buf, dtype=np.float32)
    )).shuffle(2048).batch(MINI_BATCH_SIZE)

    for _ in range(TRAIN_EPOCHS):
        for obs_b, act_b, logp_old_b, adv_b, ret_b in dataset:
            with tf.GradientTape() as tape_pi, tf.GradientTape() as tape_v:
                # Get new action probs
                pi = actor(obs_b)
                action_probs = tf.reduce_sum(pi * tf.one_hot(act_b, n_actions), axis=1)
                logp_new = tf.math.log(action_probs + 1e-8)

                # PPO loss
                loss_pi = ppo_loss(logp_old_b, logp_new, adv_b)

                # Value loss
                v_preds = critic(obs_b)[:, 0]
                loss_v = tf.reduce_mean((ret_b - v_preds) ** 2)

            grads_pi = tape_pi.gradient(loss_pi, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(grads_pi, actor.trainable_variables))

            grads_v = tape_v.gradient(loss_v, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads_v, critic.trainable_variables))


for epoch in range(50):
    print(f"Epoch {epoch+1}")
    train()
