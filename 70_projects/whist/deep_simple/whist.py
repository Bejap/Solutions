import whist_game as wg
from tqdm import tqdm
import numpy as np
from simple_whist_DQN import DQNAgent

EPISODES = 100

epsilon = 1
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.001


class Whist:
    def __init__(self, player_names: list):
        self.deck = wg.Deck()
        self.players = [wg.Player(name) for name in player_names]
        self.cards_array = [0] * 13
        self.round_array = [0] * 13
        self.hand_array = [0] * 13
        self.player_array = [0] * 4
        self.score_array = [0] * 4
        self.count = 0
        self.step_count = 0
        self.round_list = []
        self.player1_cards = [0] * 13
        self.player2_cards = [0] * 13
        self.player3_cards = [0] * 13
        self.player4_cards = [0] * 13

    def deal_cards(self):
        self.deck.shuffle()
        cards_per_player = len(self.deck.get_deck()) // len(self.players)  # CHANGE THIS

        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)

        # self.deck = [
        #     wg.Card('Hearts', 'Jack'), wg.Card('Hearts', '3'), wg.Card('Hearts', '4'),
        #     wg.Card('Hearts', '5'), wg.Card('Hearts', '6'), wg.Card('Hearts', '10'),
        #     wg.Card('Hearts', '8'), wg.Card('Hearts', 'Queen'), wg.Card('Hearts', '7'),
        #     wg.Card('Hearts', '2'), wg.Card('Hearts', '9'), wg.Card('Hearts', 'King')
        # ]
        #
        #
        # for i, player in enumerate(self.players):
        #     player.hand = self.deck[i * 3:(i + 1) * 3]

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

        init_state = self._get_init_state()
        return init_state

    def _get_init_state(self):
        current_player = self.players[(self.count - 1) % 4]
        self.hand_array = self._player_hand(current_player)
        self.player_array = [0] * 4
        self.player_array[(self.count % 4) - 1] = 1

        self._initialize_player_card_tracking()

        game_state = ([self.cards_array] + [self.round_array] + [self.hand_array]
                      + [self.player_array] + [self.score_array]
                      + [self.player1_cards] + [self.player2_cards]
                      + [self.player3_cards] + [self.player4_cards])

        return game_state

    def get_player_hand(self):
        return {player.name: player.get_hand() for player in self.players}

    def _initialize_player_card_tracking(self):
        # Reset all player card arrays
        player_card_arrays = [self.player1_cards, self.player2_cards,
                              self.player3_cards, self.player4_cards]

        for arr in player_card_arrays:
            for i in range(13):
                arr[i] = 0

        # For the current player, we know their hand exactly
        current_player_idx = self.count % 4
        current_player = self.players[current_player_idx]
        current_player_array = player_card_arrays[current_player_idx]

        for card in current_player.hand:
            card_position = card.rank_value - 2
            current_player_array[card_position] = 1

        # For cards that have been played, mark them as impossible (0)
        for i, played in enumerate(self.cards_array):
            if played == 1:
                for arr in player_card_arrays:
                    arr[i] = 0

    def step(self, action):
        done = False
        current_player = self.players[self.count % 4]

        if not current_player.hand:
            return None, 0, True

        card = current_player.action(action)
        self.round_list.append(card)
        current_player.hand.remove(card)
        # print(f"Player {current_player.name} played {card}. Count: {self.count}")

        reward = [0] * 4  # Track rewards for all players
        # print(f"{current_player} played {card}")
        # print(game_state)

        if len(self.round_list) == 4:
            trick_winner = self._evaluate_trick_winner()
            winner_index = self.players.index(trick_winner)

            # Assign rewards
            for i, player in enumerate(self.players):
                if i == winner_index:
                    reward[i] += 20  # Reward for winning the trick

                else:
                    reward[i] -= 5  # Penalty for losing the trick

            # print(f"Trick completed. Winner: Player {winner_index}. Rewards: {reward}")

            # Reset round list for next trick
            self.round_list = []

        game_state = self._get_game_state(card)
        if all(len(player.hand) == 0 for player in self.players):
            done = True
        self.step_count += 1
        # agent.train(terminal, self.step_count)

        return game_state, reward, done

    def _get_game_state(self, card: wg.Card):
        self.count += 1

        # Reset round_array every 4th card
        if self.count % 4 == 1:  # When a new trick starts
            self.round_array = [0] * 13
        if self.count % 12 == 1:
            self.cards_array = [0] * 13
            self.score_array = [0] * 4
        self.cards_array = self._cards_played(card)
        self.round_array = self._round_cards_played(card)
        current_player = self.players[(self.count - 1) % 4]
        self.hand_array = self._player_hand(current_player)
        self.player_array = [0] * 4
        self.player_array[(self.count % 4) - 1] = 1
        self._update_player_card_tracking(card)

        game_state = ([self.cards_array] + [self.round_array] + [self.hand_array]
                      + [self.player_array] + [self.score_array]
                      + [self.player1_cards] + [self.player2_cards]
                      + [self.player3_cards] + [self.player4_cards])

        return game_state

    def _update_player_card_tracking(self, card: wg.Card):
        player_idx = (self.count - 1) % 4
        card_position = card.rank_value - 2

        # Mark the card as played (impossible) for all players
        player_card_arrays = [self.player1_cards, self.player2_cards,
                              self.player3_cards, self.player4_cards]

        for arr in player_card_arrays:
            arr[card_position] = 0

    def _cards_played(self, card_s: wg.Card):
        # print(self.count)
        if card_s.rank_value is None:
            print("card_s.rank_value er None!")
            return
        card_position_s = card_s.rank_value - 2
        try:
            self.cards_array[card_position_s] = 1
        except IndexError:
            print(f"Index {card_position_s} er uden for grænserne for cards_array.")
        except TypeError:
            print(f"Ugyldig type for rank_value: {type(card_s.rank_value)}")

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

    def _evaluate_trick_winner(self):
        trick_cards = self.round_list
        print(trick_cards)

        # Find det højeste kort i farven
        winning_card = max(
            (card for card in trick_cards),
            key=lambda c: c.rank_value
        )

        winner_index = trick_cards.index(winning_card)  # Find vinderen
        winner = self.players[winner_index]

        # Opdater score_array
        self.score_array[winner_index] += 1

        return winner


player_names = ['1', '2', '3', '4']
game = Whist(player_names)
# game.play()


# for name, hand in game.get_player_hand().items():
#     print(name, hand)
agents = [DQNAgent(13 + 13 + 13 + (13 * 4) + 4 + 4) for _ in range(4)]

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    print(f'Episode: {episode + 1}/{EPISODES}')
    print(epsilon)
    count = 0

    game.reset()
    episode_rewards = [0, 0, 0, 0]
    done = False

    while count <= 12:
        pending_transitions = []
        for _ in range(4):
            current_player_index = count % 4
            current_player = game.players[current_player_index]
            agent = agents[current_player_index]  # Hent den rigtige agent

            if 0 <= count < 4:
                action_space = 3
            elif 4 <= count < 8:
                action_space = 2
            else:
                action_space = 1

            valid_actions = [i for i, value in enumerate(game._player_hand(current_player)) if value == 1]
            # print(valid_actions)

            if np.random.random() > epsilon:
                qs = agent.get_qs(current_state)
                if valid_actions:
                    # Ensure that all card values are within the valid index range of qs
                    valid_q_values = [qs[card] for card in valid_actions if card < len(qs)]

                    if valid_q_values:
                        action = np.argmax(valid_q_values)
                        # Get the corresponding card
                    else:
                        action = np.random.randint(action_space)  # Fallback in case of an issue
                else:
                    action = 0  # Default action
            else:
                if valid_actions:
                    action = np.random.randint(action_space)  # Pick a random valid card
                else:
                    action = 0  # Default action

            new_state, rewards, done = game.step(action)
            if rewards != 0:  # Check if rewards is not None
                episode_rewards[current_player_index] += rewards[current_player_index]

            # Optionally train after each step

            if new_state is not None:
                current_state = new_state  # Update state
            pending_transitions.append((current_state, action, None, new_state, False))
            # print(pending_transitions)

            if len(game.round_list) == 0:  # Trick is complete
                for i, (s, a, _, ns, _) in enumerate(pending_transitions):
                    if rewards != 0:
                        reward_value = rewards[i]  # Get reward for this player
                        # Now add to replay memory with correct reward
                        agents[i].update_replay_memory((s, a, reward_value, ns, done))

                for agent_idx, agent in enumerate(agents):
                    agent.train(done, count)

                pending_transitions = []

            count += 1

            if done:
                break

    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)  # Decay epsilon

    for agent in agents:
        agent.train(True, count)

for i, agent in enumerate(agents):
    agent.save_agent(f"agent_player_{i}.weights.h5")
    agent.save_full_agent(f"full_agent_player_{i}.keras")
