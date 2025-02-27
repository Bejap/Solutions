import numpy as np
import tensorflow as tf

state_size = 179
num_actions = 52

state = [
    player_hand,
    current_trick,
    played_cards,
    bid,
    tricks_won
]


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_actions)
])

model = create_model()

model.summary()
