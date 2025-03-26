import whist_game as wg
from tqdm import tqdm
import numpy as np
from simple_whist_DQN import DQNAgent

EPISODES = 500

epsilon = 1
EPSILON_DECAY = 0.9985
MIN_EPSILON = 0.001


class Whist:
    def __init__(self, player_names):
        self.deck = wg.Deck()
        self.players = [wg.Player(name) for name in player_names]
        self.cards_array = [0] * 13
        self.round_array = [0] * 13
        self.hand_array = [0] * 13
        self.player_array = [0] * 4
        self.score_array = [0] * 4
        self.count = 0
        self.step_count = 0

    def deal_cards(self):
        self.deck.shuffle()
        cards_per_player = len(self.deck.get_deck()) // len(self.players)  # CHANGE THIS

        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)

    def play(self):
        global game_state
        self.deal_cards()
        for player in self.players:
            card = player.action(0)
            print(card)

            game_state = self._get_game_state(card)

        return game_state

    def reset(self):
        self.deck = wg.Deck()
        self.deck.shuffle()

        for player in self.players:
            player.hand = []

        self.deal_cards()
        initial_state = self._get_game_state(self.players[0].hand[0])
        return initial_state

    def get_player_hand(self):
        return {player.name: player.get_hand() for player in self.players}

    def step(self, action):
        current_player = self.players[self.count % 4]

        if not current_player.hand:
            return None, 0, True

        card = current_player.action(action)
        game_state = self._get_game_state(card)


        trick_winner = None
        reward = 0
        print(f"{current_player} played {card}")
        print(game_state)
        current_player.hand.remove(card)

        if self.count % 4 == 0 and self.count > 0:
            trick_winner = self._evaluate_trick_winner(card)
            if trick_winner == current_player:
                reward = 20
            else:
                reward = -5

        self.step_count += 1
        return game_state, reward, trick_winner is not None

    def _get_game_state(self, card):
        self.count += 1

        # Reset round_array every 4th card
        if self.count % 4 == 1:  # When a new trick starts
            self.round_array = [0] * 13
        if self.count % 12 == 1:
            self.cards_array = [0] * 13
        self.cards_array = self._cards_played(card)
        self.round_array = self._round_cards_played(card)
        current_player = self.players[(self.count - 1) % 4]
        self.hand_array = self._player_hand(current_player)
        self.player_array = [0] * 4
        self.player_array[(self.count % 4) - 1] = 1

        game_state = ([self.cards_array] + [self.round_array] + [self.hand_array]
                      + [self.player_array] + [self.score_array])

        return game_state
    def _cards_played(self, card_s: wg.Card):
        print(self.count)
        card_position_s = card_s.rank_value - 2
        self.cards_array[card_position_s] = 1
        return self.cards_array

    def _round_cards_played(self, card_p: wg.Card):
        card_position_p = card_p.rank_value - 2
        self.round_array[card_position_p] = 1
        return self.round_array

    def _player_hand(self, player: wg.Player):
        self.hand_array = [0] * 13  # Reset hand array
        for card in player.hand:
            card_position = card.rank_value - 2
            self.hand_array[card_position] = 1
        return self.hand_array

    def _evaluate_trick_winner(self, card):
        trick_players = []

        for i in range(4):
            player_index = (self.step_count - 4 + i) % 4
            trick_players.append(player_index)
        print()
        print()


player_names = ['1', '2', '3', '4']
game = Whist(player_names)
# game.play()


# for name, hand in game.get_player_hand().items():
#     print(name, hand)
agent = DQNAgent(13 + 13 + 13 + 4 + 4)

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    print(f'Episode: {episode + 1}/{EPISODES}')
    print(epsilon)
    count = 0

    current_state = game.reset()
    episode_reward = 0
    done = False
    while len(game.players[3].hand) > 0 and count < 12:
        for _ in range(4):
            if 0 <= count < 4:
                action_space = 3
            elif 4 <= count < 8:
                action_space = 2
            else:
                action_space = 1

            current_player = game.players[game.count % 4]

            valid_actions = [i for i, value in enumerate(game._player_hand(current_player)) if value == 1]
            print(valid_actions)

            if np.random.random() > epsilon:
                # Choose the best action among valid actions
                qs = agent.get_qs(current_state)
                for valid in range(len(valid_actions)):
                    action = max(valid, key=lambda x: qs[x]) if valid_actions else 0
            else:
                # Choose a random valid action
                action = np.random.choice(len(valid_actions)) if valid_actions else 0


            new_state, reward, done = game.step(action)
            episode_reward += reward  # Track total reward

            transition = (current_state, action, reward, new_state, done)
            agent.update_replay_memory(transition)

            # Optionally train after each step
            agent.train(terminal_state=done, step=game.step_count)

            current_state = new_state  # Update state
            count += 1

    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)  # Decay epsilon

