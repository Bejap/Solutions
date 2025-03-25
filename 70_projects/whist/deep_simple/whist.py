import whist_game as wg
from tqdm import tqdm
import numpy as np

EPISODES = 500

epsilon = 1
EPSILON_DECAY = 0.9985
MIN_EPSILON = 0.001

class Whist:
    def __init__(self, player_names):
        self.deck = wg.Deck()
        self.players = [wg.Player(name) for name in player_names]

    def deal_cards(self):
        self.deck.shuffle()
        cards_per_player = len(self.deck.get_deck()) // len(self.players)  # CHANGE THIS

        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)

    def play(self):
        self.deal_cards()
        game_state = {player.name: player.get_hand() for player in self.players}
        return game_state

    def reset(self):
        self.deck = wg.Deck()
        self.deck.shuffle()

        for player in self.players:
            player.hand = []

        self.deal_cards()
        game_state = {player.name: player.get_hand() for player in self.players}
        return game_state

    def get_player_hand(self):
        return {player.name: player.get_hand() for player in self.players}


player_names = ['1', '2', '3', '4']
game = Whist(player_names)
game.play()

for name, hand in game.get_player_hand().items():
    print(name, hand)

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    print(f'Episode: {episode +1}/{EPISODES}')
    print(epsilon)

    episode_reward = 0
    step = 1

    current_state = game.reset()
    done = False
    while not done:
        if np.random.random() > epsilon:
            # action = np.argmax(agent.get_qs(current_state))
            exit
        else:
            action = np.random.randint(0, wg.Player.ACTION_SPACE)


