import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Linear_QNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.model = Sequential([
            Dense(hidden_size, activation='relu', input_shape=(input_size,)),
            Dense(output_size)
        ])

    def call(self, x):
        return self.model(x)

    def save_model(self, file_name='model.h5'):
        self.model.save(file_name)


import numpy as np

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.count = 0

    def train_step(self, state, action, reward, next_state, done):
        # Sikrer korrekt form
        self.count += 1
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        next_state = np.array(next_state, dtype=np.float32).reshape(1, -1)
        action = np.array(action, dtype=np.int32).reshape(-1)  # Gør action til en liste af integers
        reward = np.array(reward, dtype=np.float32).reshape(-1)  # Sikrer korrekt shape
        done = np.array(done, dtype=np.bool_).reshape(-1)  # Sikrer korrekt shape
        print(f"State shape: {state.shape} | Next state shape: {next_state.shape}")
        if state.shape[1] != 11:
            print("Fejl! Forkert state:", state, self.count)
            exit()

        with tf.GradientTape() as tape:
            pred = self.model(state, training=True)
            target = pred.numpy()  # Konverter til NumPy

            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    next_pred = self.model(next_state[idx].reshape(1, -1), training=False)
                    Q_new = reward[idx] + self.gamma * np.max(next_pred.numpy())

                target[idx, action[idx]] = Q_new  # Rettet indeks

            loss = self.loss_fn(target, pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))





