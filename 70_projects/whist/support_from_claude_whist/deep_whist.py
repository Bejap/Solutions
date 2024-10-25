import tensorflow as tf
import numpy as np
from collections import deque
import random


class WhistStateEncoder:
    def encode_state(self, player_hand, played_cards, current_trick, trump_suit):
        """
        Encode game state into numpy array format:
        - 52 values for player's hand (one-hot)
        - 52 values for played cards history
        - 52 values for current trick
        - 4 values for trump suit (one-hot)
        - 152 values for void suits tracking
        """
        state = np.zeros(312)
        # Encode game state logic here...
        return state.astype(np.float32)


class WhistModel(tf.keras.Model):
    def __init__(self):
        super(WhistModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.layer3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(52)  # Q-values for each card

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output_layer(x)


class WhistAI:
    def __init__(self):
        # Main network and target network
        self.model = WhistModel()
        self.target_model = WhistModel()

        # Training parameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor

        self.encoder = WhistStateEncoder()

        # Initialize target network with same weights
        self.update_target_network()

    @tf.function
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and tf.random.uniform(()) < self.epsilon:
            return tf.random.uniform((), 0, 52, dtype=tf.int32)

        state_tensor = tf.expand_dims(state, 0)
        q_values = self.model(state_tensor)
        return tf.argmax(q_values[0])

    def update_memory(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Single training step using gradient tape"""
        with tf.GradientTape() as tape:
            # Current Q Values
            current_q = self.model(states)
            current_q_values = tf.gather(current_q, actions, batch_dims=1)

            # Target Q Values
            next_q = self.target_model(next_states)
            max_next_q = tf.reduce_max(next_q, axis=1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q

            # Compute loss
            loss = self.loss_fn(target_q_values, current_q_values)

        # Apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Perform training step
        loss = self.train_step(states, actions, rewards, next_states, dones)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss

    def update_target_network(self):
        """Update target network weights"""
        self.target_model.set_weights(self.model.get_weights())


def train_ai():
    """Training loop"""
    ai = WhistAI()
    num_episodes = 10000

    for episode in range(num_episodes):
        game = WhistGame()  # Your game environment
        state = game.get_initial_state()
        total_reward = 0
        done = False

        while not done:
            # Get legal moves
            legal_moves = game.get_legal_moves()

            # Encode state
            encoded_state = ai.encoder.encode_state(
                state['hand'],
                state['played_cards'],
                state['current_trick'],
                state['trump_suit']
            )

            # Choose action
            action = int(ai.get_action(encoded_state))

            # Ensure action is legal
            if action not in legal_moves:
                action = random.choice(legal_moves)

            # Perform action
            next_state, reward, done = game.step(action)

            # Encode next state
            encoded_next_state = ai.encoder.encode_state(
                next_state['hand'],
                next_state['played_cards'],
                next_state['current_trick'],
                next_state['trump_suit']
            )

            # Store experience
            ai.update_memory(encoded_state, action, reward, encoded_next_state, done)

            # Train network
            loss = ai.train()

            state = next_state
            total_reward += reward

        # Update target network periodically
        if episode % 100 == 0:
            ai.update_target_network()
            print(f"Episode {episode}, Total Reward: {total_reward}")


# Custom metrics tracking
@tf.keras.utils.register_keras_serializable()
class WhistMetrics(tf.keras.metrics.Metric):
    def __init__(self, name='whist_metrics', **kwargs):
        super(WhistMetrics, self).__init__(name=name, **kwargs)
        self.tricks_won = self.add_weight(name='tricks_won', initializer='zeros')
        self.games_won = self.add_weight(name='games_won', initializer='zeros')