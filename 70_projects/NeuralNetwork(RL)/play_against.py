import numpy as np
from tensorflow.keras.models import load_model

class KasinoEnv:
    def __init__(self):
        self.player_hand = np.random.randint(1, 14, size=4)
        self.table_cards = np.random.randint(1, 14, size=4)
        self.state = np.concatenate([self.player_hand, self.table_cards])

    def reset(self):
        self.__init__()
        return self.state

    def step(self, action):
        card_played = self.player_hand[action]
        reward = 0

        if card_played in self.table_cards:
            reward = 1
            self.table_cards = np.delete(self.table_cards, np.where(self.table_cards == card_played))
        else:
            reward = -1

        self.player_hand[action] = 0

        while len(self.table_cards) < 4:
            self.table_cards = np.append(self.table_cards, 0)

        self.state = np.concatenate([self.player_hand, self.table_cards])
        done = np.all(self.player_hand == 0)
        return self.state, reward, done, {}

# Load trained model
trained_model = load_model("kasino_rl_model.h5")

# Initialize environment
env = KasinoEnv()
state = env.reset()

print(f"Your hand: {env.player_hand}")
print(f"Table cards: {env.table_cards}")

done = False
while not done:
    # Human turn
    print(f"Your hand: {env.player_hand}")
    action = int(input("Choose a card to play (0-3): "))
    state, reward, done, _ = env.step(action)
    print(f"Table cards: {env.table_cards}")
    print(f"Reward: {reward}")
    if done:
        print("Game over!")
        break

    # AI turn
    state = np.reshape(state, [1, 8])
    action = np.argmax(trained_model.predict(state, verbose=0))
    print(f"AI plays card at index {action}")
    state, reward, done, _ = env.step(action)
    print(f"Table cards: {env.table_cards}")
    print(f"Reward: {reward}")
    if done:
        print("Game over!")
