import numpy as np
from perplex2 import Player, Deck, Card
from perplex3 import DQNAgent
import random

class GameState:
    def __init__(self, num_players=4, array_length=13):
        self.array_length = array_length
        self.num_players = num_players
        self.reset()
        self.agent = DQNAgent(input_size=self._get_state_size())

    def reset(self):
        self.cards_played = np.zeros(self.array_length, dtype=int)
        self.current_trick = np.zeros(self.array_length, dtype=int)
        self.player_turn = np.zeros(self.num_players, dtype=int)
        self.scores = np.zeros(self.num_players, dtype=int)
        self.player_hands = [np.zeros(self.array_length, dtype=int)
                             for _ in range(self.num_players)]
        self.known_actions = [[] for _ in range(self.num_players)]

    def update_from_game(self, game):
        """Convert game object to state representation"""
        # Update cards
        for i, player in enumerate(game.players):
            self.player_hands[i] = self._hand_to_array(player.hand)

        # Update scores
        self.scores = [p.tricks_won for p in game.players]

    def _hand_to_array(self, hand):
        arr = np.zeros(self.array_length, dtype=int)
        for card in hand:
            arr[card.rank_value - 2] = 1
        return arr

    def get_state(self, player_id):
        """Get observation space for specific player"""
        return {
            'global_cards': self.cards_played,
            'current_trick': self.current_trick,
            'player_hand': self.player_hands[player_id],
            'scores': self.scores,
            'turn_indicator': self.player_turn,
            'known_actions': self.known_actions[player_id]
        }

    def _get_state_size(self):
        return (self.state.array_length * 2  # cards_played + current_trick
                + self.state.array_length    # player_hand
                + 4                         # scores
                + 4                         # turn_indicator
                + self.state.array_length)

    def _state_to_model_input(self, state):
        # Convert state dict to model's expected format
        return [
            state['global_cards'],
            state['current_trick'],
            state['player_hand'],
            state['scores'],
            state['turn_indicator'],
            state['known_actions']
        ]


class WhistGame:
    def __init__(self, num_players=4):
        self.players = [Player(i + 1) for i in range(num_players)]
        self.deck = Deck()
        self.current_trick = []
        self.trump_suit = None
        self.lead_suit = None
        self.turn_order = 0
        self.trump_suit = random.choice(list(Card.SUIT_VALUES.keys()))

    def new_round(self):
        self.deck = Deck()
        self.deck.shuffle()
        self._deal_cards()
        self.current_trick = []
        self.lead_suit = None

    def _deal_cards(self):
        cards_per_player = len(self.deck.get_deck()) // len(self.players)
        for player in self.players:
            player.hand = self.deck.deal(cards_per_player)
            player.sort_hand()

    def play_card(self, player_id, card_idx):
        player = self.players[player_id]
        if not self._validate_move(player, card_idx):
            raise InvalidMoveError()

        played_card = player.play_card(card_idx)
        self.current_trick.append((player_id, played_card))

        if len(self.current_trick) == 1:
            self.lead_suit = played_card.suit

        if len(self.current_trick) == 4:
            winner_id = self._determine_trick_winner()
            self.players[winner_id].tricks_won += 1
            return True, winner_id

        return False, None

    def _validate_move(self, player, card_idx):
        if self.lead_suit is None:  # First player can play any card
            return True
        if player.hand[card_idx].suit == self.lead_suit:
            return True
        return not any(c.suit == self.lead_suit for c in player.hand)

    def _determine_trick_winner(self):
        # Proper trick resolution with trump suit
        leading_suit = self.current_trick[0][1].suit
        trump_cards = [c for _, c in self.current_trick if c.suit == self.trump_suit]

        if trump_cards:
            return max((c for c in self.current_trick if c[1].suit == self.trump_suit),
                       key=lambda x: x[1].rank_value)[0]
        else:
            return max((c for c in self.current_trick if c[1].suit == leading_suit),
                       key=lambda x: x[1].rank_value)[0]


class GameManager:
    def __init__(self):
        self.game = WhistGame()
        self.state = GameState()

    def reset(self):
        self.game.new_round()
        self.state.reset()
        self.state.update_from_game(self.game)
        return self.state.get_state(0)  # For first player

    def step(self, player_id, action):
        trick_completed, winner = self.game.play_card(player_id, action)
        self.state.update_from_game(self.game)

        reward = self._calculate_reward(player_id, trick_completed, winner)
        done = self._check_game_over()

        return self.state.get_state(player_id), reward, done

    def _calculate_reward(self, player_id, completed, winner):
        if not completed:
            return 0
        return 20 if winner == player_id else -5

    def _check_game_over(self):
        return all(len(p.hand) == 0 for p in self.game.players)

    def train(self, episodes=1000):
        epsilon = 1.0  # Exploration rate
        epsilon_min = 0.01
        epsilon_decay = 0.995

        for episode in range(episodes):
            state = self.reset()
            done = False
            current_player = 0

            while not done:
                # Get action from agent
                if random.random() < epsilon:
                    action = random.randint(0, len(self.game.players[current_player].hand) - 1)
                else:
                    processed_state = self._state_to_model_input(state)
                    action = np.argmax(self.agent.get_qs(processed_state))

                # Take action
                next_state, reward, done = self.step(current_player, action)

                # Store experience
                self.agent.update_replay_memory((
                    self._state_to_model_input(state),
                    action,
                    reward,
                    self._state_to_model_input(next_state),
                    done
                ))

                # Train agent
                self.agent.train(done)

                # Update state
                state = next_state
                current_player = (current_player + 1) % 4

            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)


manager = GameManager()
state = manager.reset()

while not done:
    action = agent.choose_action(state)
    next_state, reward, done = manager.step(player_id, action)
    agent.learn(state, action, reward, next_state)
    state = next_state
