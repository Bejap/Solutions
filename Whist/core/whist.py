from Whist.core import whist_game as wg
import numpy as np
import random
import logging
from Whist.utils.base_classes import BaseGame
from Whist.utils.constants import (
    LEGACY_EPISODES, 
    LEGACY_EPSILON_DECAY, 
    DEFAULT_MIN_EPSILON, 
    ARRAY_LENGTH, 
    NUM_PLAYERS,
    CARDS_PER_PLAYER,
    ENABLE_PER_CARD_REWARD,
    PER_CARD_EW_STRATEGY_REWARD,
    PER_CARD_EW_STRATEGY_PENALTY,
    END_GAME_REWARD_MULTIPLIER
)

# Configure logger for reward monitoring
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

EPISODES = LEGACY_EPISODES

epsilon = 1
EPSILON_DECAY = LEGACY_EPSILON_DECAY
MIN_EPSILON = DEFAULT_MIN_EPSILON


class Whist(BaseGame):
    def __init__(self, player_names: list, enable_per_card_reward: bool = ENABLE_PER_CARD_REWARD):
        super().__init__(NUM_PLAYERS)
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
        self.next_player_is_winner = None  # Track if next player should be trick winner
        
        # Per-card reward configuration
        self.enable_per_card_reward = enable_per_card_reward
        self.per_card_reward = PER_CARD_EW_STRATEGY_REWARD
        self.per_card_penalty = PER_CARD_EW_STRATEGY_PENALTY
        
        # Store EW strategy reference for per-card rewards (set externally)
        self.ew_strategies = None
        
        # Monitoring: Track reward statistics per episode
        self.reward_stats = {
            'agent_0_total': 0,
            'agent_2_total': 0,
            'agent_0_wins': 0,
            'agent_2_wins': 0,
            'tricks_completed': 0,
            'per_card_rewards': 0  # Track per-card rewards given
        }
    
    def set_monitoring_level(self, level='INFO'):
        """Set the logging level for monitoring reward system.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        logger.setLevel(getattr(logging, level))
        logger.info(f"Monitoring level set to {level}")
    
    def get_reward_stats(self):
        """Get current reward statistics for monitoring."""
        return self.reward_stats.copy()
    
    def reset_reward_stats(self):
        """Reset reward statistics at the start of a new episode."""
        self.reward_stats = {
            'agent_0_total': 0,
            'agent_2_total': 0,
            'agent_0_wins': 0,
            'agent_2_wins': 0,
            'tricks_completed': 0,
            'per_card_rewards': 0
        }
    
    def set_ew_strategies(self, ew_strategies: dict):
        """Set EW strategy references for per-card reward calculation.
        
        Args:
            ew_strategies: Dictionary mapping player index to EWStrategy object
        """
        self.ew_strategies = ew_strategies
    
    def calculate_per_card_reward(self, player_idx: int, action: int, valid_actions: list) -> float:
        """Calculate per-card reward based on EW strategy matching.
        
        This rewards agents for playing cards that match what EW strategy would play.
        This feature can be disabled by setting enable_per_card_reward=False.
        
        Args:
            player_idx: Index of the current player (0-3)
            action: The action (card index) the agent chose
            valid_actions: List of valid action indices
            
        Returns:
            Reward value (positive if matching EW strategy, negative otherwise)
        """
        if not self.enable_per_card_reward or self.ew_strategies is None:
            return 0.0
        
        # Only apply per-card rewards to agent positions (0 and 2)
        if player_idx not in [0, 2]:
            return 0.0
        
        # Get what EW strategy would choose
        # We need to use the strategy for position 1 or 3 as reference
        # Use position 1 (East) as the reference strategy
        reference_strategy = self.ew_strategies.get(1)
        if reference_strategy is None:
            return 0.0
        
        current_player = self.players[player_idx]
        
        # Get EW strategy's preferred action
        try:
            ew_action = reference_strategy._strategic_action(current_player, valid_actions)
        except (IndexError, AttributeError) as e:
            # Log the error for debugging but don't crash - just return no reward
            logger.debug(f"Per-card reward calculation failed: {e}")
            return 0.0
        
        # Reward if agent matches EW strategy
        if action == ew_action:
            self.reward_stats['per_card_rewards'] += self.per_card_reward
            return self.per_card_reward
        else:
            self.reward_stats['per_card_rewards'] += self.per_card_penalty
            return self.per_card_penalty
    
    @staticmethod
    def _get_card_position(card: wg.Card):
        """Calculate the position of a card in the state array.
        
        Maps 52 cards to positions 0-51:
        - Clubs (suit_value=0): positions 0-12 (rank 2-A)
        - Diamonds (suit_value=1): positions 13-25 (rank 2-A)
        - Hearts (suit_value=2): positions 26-38 (rank 2-A)
        - Spades/Trump (suit_value=3): positions 39-51 (rank 2-A)
        """
        return card.suit_value * 13 + (card.rank_value - 2)

    def deal_cards(self):
        self.deck.shuffle()
        # Use CARDS_PER_PLAYER from constants instead of calculating from deck size
        # This allows for smaller game variants (e.g., 9 or 11 cards per player)
        cards_per_player = CARDS_PER_PLAYER

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

    def reset(self, seed=None):
        """Reset the game state.
        
        Args:
            seed: Optional random seed for reproducible games (for benchmarking)
        """
        # Set seed if provided (for reproducible benchmark games)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.deck = wg.Deck()
        self.deck.shuffle()
        self.trick_winner = None
        self.turn_counter = 0
        self.score_array = [0] * 4
        self.next_player_is_winner = None  # Reset winner tracking
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
        
        # Reset reward statistics for new episode
        self.reset_reward_stats()

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
            card_position = self._get_card_position(card)
            current_player_array[card_position] = card.rank_value
            # print(current_player_array)

        # For cards that have been played, mark them as impossible (0)
        for i, played in enumerate(self.cards_array):
            if played == 1:
                for arr in player_card_arrays:
                    arr[i] = 0

    def get_valid_actions(self, player):
        """Get valid card indices for a player based on follow suit rules.
        
        Args:
            player: The player whose valid actions to get
            
        Returns:
            List of valid card indices (positions in sorted hand)
        """
        player._sort_hand()
        
        # If this is the first card of the trick, all cards are valid
        if len(self.round_list) == 0:
            return list(range(len(player.hand)))
        
        # Get the led suit (first card played in the trick)
        led_suit = self.round_list[0][1].suit
        
        # Find cards of the led suit
        led_suit_indices = [i for i, card in enumerate(player.hand) if card.suit == led_suit]
        
        # If player has cards of the led suit, they must play one of them
        if led_suit_indices:
            return led_suit_indices
        
        # If player doesn't have led suit, they can play any card
        return list(range(len(player.hand)))
    
    def step(self, action):
        done = False
        current_player = self.players[self.current_player_idx]

        if not current_player.hand:
            return None, 0, True

        card = current_player.action(action)
        self.round_list.append((self.current_player_idx, card))
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
            # New reward system: based on (tricks_won - max_tricks)
            # max_tricks for each player is CARDS_PER_PLAYER
            max_tricks = CARDS_PER_PLAYER
            
            # Calculate end-game reward for each agent based on their individual tricks won
            agent_0_tricks = self.score_array[0]
            agent_2_tricks = self.score_array[2]
            
            # Reward is (tricks_won - max_tricks) * multiplier
            # This gives a range from -max_tricks to 0 (since max possible is max_tricks)
            agent_0_end_reward = (agent_0_tricks - max_tricks) * END_GAME_REWARD_MULTIPLIER
            agent_2_end_reward = (agent_2_tricks - max_tricks) * END_GAME_REWARD_MULTIPLIER
            
            reward[0] += agent_0_end_reward
            reward[2] += agent_2_end_reward
            self.reward_stats['agent_0_total'] += agent_0_end_reward
            self.reward_stats['agent_2_total'] += agent_2_end_reward
            
            team_1_score = self.score_array[0] + self.score_array[2]  # Agents' team
            team_2_score = self.score_array[1] + self.score_array[3]  # Opponents' team
            
            logger.info(f"Game ended. Team 1 Score: {team_1_score}, Team 2 Score: {team_2_score}. "
                       f"Agent 0 end reward: {agent_0_end_reward:.1f} (tricks: {agent_0_tricks}/{max_tricks}), "
                       f"Agent 2 end reward: {agent_2_end_reward:.1f} (tricks: {agent_2_tricks}/{max_tricks})")

        # Update current_player_idx
        # If a trick was just completed, the winner should lead the next trick
        if self.next_player_is_winner is not None:
            self.current_player_idx = self.next_player_is_winner
            self.next_player_is_winner = None
        else:
            # Normal increment for cards within a trick
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

            # Assign rewards only to the 2 agents (positions 0 and 2)
            # Positions 1 and 3 use strategic rule-based play and don't need rewards
            agent_positions = [0, 2]  # North and South are the DQN agents (Team 1)
            
            for i in agent_positions:
                if i == winner_index:
                    # Agent wins the trick: +1
                    reward[i] = 1
                    # Update monitoring stats
                    if i == 0:
                        self.reward_stats['agent_0_wins'] += 1
                        self.reward_stats['agent_0_total'] += 1
                    else:  # i == 2
                        self.reward_stats['agent_2_wins'] += 1
                        self.reward_stats['agent_2_total'] += 1
                elif winner_index in agent_positions:
                    # Partner wins the trick: +0.8
                    reward[i] = 0.8
                    # Update monitoring stats
                    if i == 0:
                        self.reward_stats['agent_0_total'] += 0.8
                    else:  # i == 2
                        self.reward_stats['agent_2_total'] += 0.8
                else:
                    # Opponent (non-agent) wins the trick: -1
                    reward[i] = -1
                    # Update monitoring stats
                    if i == 0:
                        self.reward_stats['agent_0_total'] -= 1
                    else:  # i == 2
                        self.reward_stats['agent_2_total'] -= 1
            
            self.reward_stats['tricks_completed'] += 1
            
            # Log reward distribution for monitoring
            logger.debug(f"Trick {self.reward_stats['tricks_completed']} completed. Winner: Player {winner_index}. "
                        f"Agent rewards: Agent 0: {reward[0]}, Agent 2: {reward[2]}")

            # Reset round list for next trick
            self.round_list = []
            
            # IMPORTANT: Winner of the trick leads the next trick
            # Set current_player_idx to winner so they start next trick
            # Note: step() will increment it by 1, so we need to set it to winner-1
            # But we need to handle this after the step increment
            self.next_player_is_winner = winner_index

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
            card_pos = self._get_card_position(played_card)
            if player_arrays[player_id][card_pos] != 0:
                player_arrays[player_id][card_pos] = 1  # Mark as seen

    def _cards_played(self, card_s: wg.Card):
        if card_s.rank_value is None:
            return
        card_position_s = self._get_card_position(card_s)
        try:
            self.cards_array[card_position_s] = 1
        except (IndexError, TypeError):
            pass

        return self.cards_array

    def _round_cards_played(self, card_p: wg.Card):
        card_position_p = self._get_card_position(card_p)
        self.round_array[card_position_p] = card_p.rank_value
        return self.round_array

    def player_hand(self, player: wg.Player):
        self.hand_array = [0] * ARRAY_LENGTH  # Reset hand array
        for card in player.hand:
            card_position = self._get_card_position(card)
            self.hand_array[card_position] = card.rank_value

        if not hasattr(self, 'static_hands'):
            self.static_hands = {}

        if player.id not in self.static_hands:
            self.static_hands[player.id] = self.hand_array.copy()

        # print("hand", self.hand_array)

        return self.hand_array

    def _evaluate_trick_winner(self):
        """Evaluate the winner of a trick considering trump (Spades).
        
        Rules:
        1. Trump (Spades) beats any non-trump card
        2. Highest trump wins if multiple trumps played
        3. If no trump, highest card in led suit wins
        """
        trick_cards = self.round_list
        
        # Get the led suit (first card played)
        led_suit = trick_cards[0][1].suit
        
        # Separate trump cards from non-trump cards
        trump_cards = [(pid, card) for pid, card in trick_cards if card.is_trump()]
        
        if trump_cards:
            # If any trump card was played, highest trump wins
            winning_tuple = max(trump_cards, key=lambda t: t[1].rank_value)
        else:
            # No trump played, highest card in led suit wins
            led_suit_cards = [(pid, card) for pid, card in trick_cards if card.suit == led_suit]
            winning_tuple = max(led_suit_cards, key=lambda t: t[1].rank_value)
        
        winner_player_id, winning_card = winning_tuple
        winner = self.players[winner_player_id]
        self.score_array[winner_player_id] += 1
        
        logger.debug(f"Trick winner: Player {winner_player_id} with {winning_card}")
        
        return winner

def reset_to_fixed(self):
        self.deck = wg.Deck()
        self.deck.shuffle()
        self.trick_winner = None
        self.turn_counter = 0
        self.score_array = [0] * 4
        self.next_player_is_winner = None  # Reset winner tracking
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
        
        # Reset reward statistics for new episode
        self.reset_reward_stats()

        init_state = self.get_init_state()
        return init_state


if __name__ == '__main__':
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)