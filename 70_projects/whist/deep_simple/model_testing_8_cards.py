import numpy as np
from tensorflow import keras
from simple_whist_DQN import DQNAgent as Dq

# Load the full model
model = keras.models.load_model('full_agent_player_0.keras')
model.summary()

# Opret en dummy-agent og erstat dens model med den indlæste model
dummy_agent = Dq(input_size=None)  # Du behøver ikke input_size her
dummy_agent.model = model  # Udskift modellen
dummy_agent.target_model = model  # Sørg for at target_model matcher også


# Forbered en test-state i korrekt format
def prepare_test_input():
    cards_array = [0, 0, 1, 1, 1, 1, 0, 1]
    round_array = [0, 0, 0, 0, 1, 0, 0, 0]
    hands_array = [0, 0, 0, 0, 0, 0, 1, 0]
    player_array = [0, 1, 0, 0]
    score_array = [0, 0, 1, 0]
    player_1_array = [0, 0, 0, 0, 0, 1, 0, 0]
    player_2_array = [0, 0, 0, 1, 0, 0, 1, 0]
    player_3_array = [0, 0, 0, 0, 0, 0, 0, 0]
    player_4_array = [0, 0, 0, 0, 0, 0, 0, 0]

    # Formatér som den forventede state-struktur
    game_state = [
        np.array(cards_array),
        np.array(round_array),
        np.array(hands_array),
        np.array(player_array),
        np.array(player_1_array),
        np.array(player_2_array),
        np.array(player_3_array),
        np.array(player_4_array),
        np.array(score_array)
    ]

    return game_state


# Brug agentens get_qs til at hente Q-værdier
state = prepare_test_input()
q_values = dummy_agent.get_qs(state)

print(f"Predicted Q-values: {q_values}")
print(f"Best action according to the model: {np.argmax(q_values)}")
