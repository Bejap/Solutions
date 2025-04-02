import numpy as np
import tensorflow as tf
from tensorflow import keras

# For .keras models
model = keras.models.load_model('full_agent_player_1.keras')
model.summary()

# For .h5 weights files (you'll need the model architecture first)
# Load a pre-defined model architecture
# You need to define this based on your model
model.load_weights('agent_player_1.weights.h5')


# self.deck = [
#             wg.Card('Hearts', 'Jack'), wg.Card('Hearts', '3'), wg.Card('Hearts', '4'),
#             wg.Card('Hearts', '5'), wg.Card('Hearts', '6'), wg.Card('Hearts', '10'),
#             wg.Card('Hearts', '8'), wg.Card('Hearts', 'Queen'), wg.Card('Hearts', '7'),
#             wg.Card('Hearts', '2'), wg.Card('Hearts', '9'), wg.Card('Hearts', 'King')
#         ]

def prepare_test_input():
    # Example state - all zeros with some test values
    cards_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # No cards played yet
    round_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # No cards in current round
    hands_array = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]  # Player has first 3 cards
    player_array = [1, 0, 0, 0]  # First player's turn
    score_array = [0, 0, 0, 0]
    player_1_array = [0] * 13
    player_2_array = [0] * 13
    player_3_array = [0] * 13
    player_4_array = [0] * 13


    # Combine all arrays into a single game state
    game_state = cards_array + round_array + hands_array+ player_array + score_array + player_1_array + player_2_array + player_3_array + player_4_array

    # Convert to numpy array and reshape for the model
    return np.array([game_state])  # Batch size of 1


# Test the model with some input
test_input = prepare_test_input()  # Create appropriate test input

# for layer in model.layers:
#     weights = layer.get_weights()
#     print(f"Layer {layer.name} weights shapes: {[w.shape for w in weights]}")

# Get the model's configuration
config = model.get_config()

predictions = model.predict(test_input)
# print(predictions)

# print(f"Model output shape: {predictions.shape}")
print(f"Predicted Q-values: {predictions[0]}")  # Show Q-values for each action

# Find the highest Q-value action
best_action = np.argmax(predictions[0])
print(f"Best action according to the model: {best_action}")
