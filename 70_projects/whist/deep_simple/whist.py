import whist_game as wg
import numpy as np

EPISODES = 250

epsilon = 1
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
ARRAY_LENGTH = 13


class Whist:
    def __init__(self, player_names: list):
        self.deck = wg.Deck()
        self.players = [wg.Player(name) for name in player_names]
        self.team_1 = [self.players[0], self.players[2]]
        self.team_2 = [self.players[1], self.players[3]]
        self.trick_winner = None
        self.current_player_idx = 0
        self.cards_array = [0] * ARRAY_LENGTH
        self.round_array = [0] * ARRAY_LENGTH
        self.hand_array = [0] * ARRAY_LENGTH
        self.player_array = [0] * 4
        self.score_array = [0] * 4
        self.count = 0
        self.static_hands = {}
        self.step_count = 0
        self.round_list = []
        self.player1_cards = [0] * ARRAY_LENGTH
        self.player2_cards = [0] * ARRAY_LENGTH
        self.player3_cards = [0] * ARRAY_LENGTH
        self.player4_cards = [0] * ARRAY_LENGTH
        self.another_count = 0
        self.turn_counter = 0

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

    def reset(self):
        self.deck = wg.Deck()
        self.deck.shuffle()
        self.trick_winner = None
        self.turn_counter = 0
        self.score_array = [0] * 4
        for player in self.players:
            player.resetting_observation()

        for player in self.players:
            player.hand = []

        self.static_hands = {}
        self.deal_cards()
        self.round_list = []
        self.current_player_idx = np.random.randint(0, 4)
        self.cards_array = [0] * ARRAY_LENGTH
        self.round_array = [0] * ARRAY_LENGTH
        self.player1_cards = [0] * ARRAY_LENGTH
        self.player2_cards = [0] * ARRAY_LENGTH
        self.player3_cards = [0] * ARRAY_LENGTH
        self.player4_cards = [0] * ARRAY_LENGTH
        self._initialize_player_card_tracking()

        init_state = self.get_init_state()
        return init_state

    def get_init_state(self):
        current_player = self.players[self.current_player_idx]
        self.hand_array = self.player_hand(current_player)
        self.player_array = [0] * 4
        if self.turn_counter % 4 == 0 and self.turn_counter > 0:  # After exactly 4 cards
            self.round_array = [0] * ARRAY_LENGTH

        self.player_array = [1 if i == self.current_player_idx else 0 for i in range(4)]

        self.player1_cards, self.player2_cards, self.player3_cards, self.player4_cards, = current_player.return_other_hand(self.static_hands[current_player.id], ARRAY_LENGTH)

        game_state = ([self.cards_array] + [self.round_array] + [self.hand_array]
                      + [self.player_array]
                      + [self.player1_cards] + [self.player2_cards]
                      + [self.player3_cards] + [self.player4_cards]
                      + [self.score_array])

        return game_state

    def _initialize_player_card_tracking(self):
        # Reset all player card arrays
        player_card_arrays = [self.player1_cards, self.player2_cards,
                              self.player3_cards, self.player4_cards]

        for arr in player_card_arrays:
            for i in range(ARRAY_LENGTH):
                arr[i] = 0

        current_player = self.players[self.current_player_idx]
        current_player_array = player_card_arrays[self.current_player_idx]

        for card in current_player.hand:
            card_position = card.rank_value - 2
            current_player_array[card_position] = card.rank_value
            # print(current_player_array)

        # For cards that have been played, mark them as impossible (0)
        for i, played in enumerate(self.cards_array):
            if played == 1:
                for arr in player_card_arrays:
                    arr[i] = 0

    def step(self, action):
        done = False
        current_player = self.players[self.current_player_idx]

        if not current_player.hand:
            return None, 0, True

        card = current_player.action(action)
        self.round_list.append((self.count % 4, card))
        self._count_cards_in_round()
        current_player.hand.remove(card)
        # print(f"Player {current_player.name} played {card}. Count: {self.count}")

        # print(f"{current_player} played {card}")
        # print(game_state)
        for player in self.players:
            player.observe(current_player.id, card.rank_value)

        game_state, reward = self._get_game_state(card)
        if all(len(player.hand) == 0 for player in self.players):
            done = True

        self.current_player_idx = (self.current_player_idx + 1) % 4

        self.step_count += 1
        if self.turn_counter % 4 == 0 and self.turn_counter > 0:  # After exactly 4 cards
            self.round_array = [0] * ARRAY_LENGTH
        self.turn_counter += 1
        # agent.train(terminal, self.step_count)

        return game_state, reward, done

    def _get_game_state(self, card: wg.Card):
        self.count += 1
        self.another_count += 1
        self.cards_array = self._cards_played(card)
        self.round_array = self._round_cards_played(card)

        current_player = self.players[self.current_player_idx]
        self.hand_array = self.player_hand(current_player)

        self.player_array = [1 if i == self.current_player_idx else 0 for i in range(4)]

        self.player1_cards, self.player2_cards, self.player3_cards, self.player4_cards, = current_player.return_other_hand(self.static_hands[current_player.id], ARRAY_LENGTH)
        # if self.turn_counter % ARRAY_LENGTH == 4:
        # self.cards_array = [0] * ARRAY_LENGTH
        # self.score_array = [0] * 4
        # for player in self.players:
        #     player.resetting_observation()

        reward = [0] * 4  # Track rewards for all players
        if len(self.round_list) == 4:
            self.trick_winner = self._evaluate_trick_winner()
            winner_index = self.players.index(self.trick_winner)

            # Assign rewards
            for i, player in enumerate(self.players):
                if i == winner_index:
                    reward[i] += 10  # Reward for winning the trick

                else:
                    reward[i] -= 5  # Penalty for losing the trick

            # print(f"Trick completed. Winner: Player {winner_index}. Rewards: {reward}")

            # Reset round list for next trick
            self.round_list = []

        # Structure the game state as a list for easier processing by the multi-input network
        game_state = [
            self.cards_array,  # [0] Cards played so far
            self.round_array,  # [1] Cards played this round
            self.hand_array,  # [2] Current player's hand
            self.player_array,  # [3] Player turn indicator
            self.player1_cards,  # [4] Player 1's possible cards
            self.player2_cards,  # [5] Player 2's possible cards
            self.player3_cards,  # [6] Player 3's possible cards
            self.player4_cards,  # [7] Player 4's possible cards
            self.score_array  # [8] Player scores
        ]

        return game_state, reward

    def _count_cards_in_round(self):
        # Index of the player who just played (last one to play)
        player_index = (self.another_count - 1) % 4

        # All four players’ card arrays
        player_arrays = [
            self.player1_cards,
            self.player2_cards,
            self.player3_cards,
            self.player4_cards
        ]

        for player_id, played_card in self.round_list:
            card_pos = played_card.rank_value - 2
            if player_arrays[player_id][card_pos] != 0:
                player_arrays[player_id][card_pos] = 1  # Mark as seen

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
        self.round_array[card_position_p] = card_p.rank_value
        return self.round_array

    def player_hand(self, player: wg.Player):
        self.hand_array = [0] * ARRAY_LENGTH  # Reset hand array
        for card in player.hand:
            card_position = card.rank_value - 2
            self.hand_array[card_position] = card.rank_value

        if not hasattr(self, 'static_hands'):
            self.static_hands = {}

        if player.id not in self.static_hands:
            self.static_hands[player.id] = self.hand_array.copy()

        # print("hand", self.hand_array)

        return self.hand_array

    def _evaluate_trick_winner(self):
        trick_cards = self.round_list
        # print(trick_cards)

        # Find det højeste kort i farven
        winning_tuple = max(
            (entry for entry in trick_cards),
            key=lambda t: t[1].rank_value  # t = (player_id, card)
        )

        winner_player_id, winning_card = winning_tuple

        winner = self.players[winner_player_id]
        self.score_array[winner_player_id ] += 1

        return winner



if __name__ == '__main__':
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)