import tensorflow as tf
from lets_try import Whist, WhistAI, Player  # Import necessary classes from your main whist script


def load_model():
    """Load the saved model for AI to use in gameplay."""
    try:
        model = tf.keras.models.load_model('../claude(need_debug).keras')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def play_with_ai():
    """Set up a Whist game with the AI using the loaded model."""
    # Load the trained model
    loaded_model = load_model()
    if loaded_model is None:
        print("Model could not be loaded. Exiting game.")
        return

    # Initialize AI and assign the loaded model
    ai = WhistAI()
    ai.model = loaded_model  # Set the loaded model as the active model for AI

    # Create a Whist game with AI and human players
    game = Whist(["Human", "AI1", "AI2", "AI3"], human_player_index=0)

    # Play the game
    winner = game.play_game(ai)

    # Display the result
    print(f"The winner is: {winner.name} with {winner.tricks_won} tricks won.")


if __name__ == "__main__":
    play_with_ai()
