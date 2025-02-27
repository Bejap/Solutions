import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy


class Card:
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}
    SUIT_VALUES = {'Clubs': 1, 'Diamonds': 2, 'Hearts': 3, 'Spades': 4}
    SUITS = list(SUIT_VALUES.keys())
    RANKS = list(RANK_VALUES.keys())

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.rank_value = self.RANK_VALUES[self.rank]
        self.suit_value = self.SUIT_VALUES[self.suit]

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __lt__(self, other):
        return self.rank_value < other.rank_value


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Card.SUIT_VALUES.keys()
                      for rank in Card.RANK_VALUES.keys()]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, player_id):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id
        self.memory = ReplayBuffer(capacity=100000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().detach().numpy()[0]

        # Filter q_values to only include valid actions
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values, key=valid_q_values.get)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]

        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calculate loss and update
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


class Player:
    def __init__(self, player_id: int, is_agent=False):
        self.player_id = player_id
        self.hand = []
        self.is_agent = is_agent

        if is_agent:
            # Initialize DQN agent with state size = 52 (cards) + 52 (round) + 52*4 (hands) + 4 (lead) + 5 (trump) + 4 (player) = 273
            # Action size = 52 (all possible cards)
            self.agent = DQNAgent(273, 52, player_id)

    def __str__(self):
        return f"Player {self.player_id + 1}: {', '.join(str(card) for card in self.hand)}"

    def get_valid_actions(self, lead_suit):
        if not lead_suit:  # Player is leading
            return [self.get_card_index(card) for card in self.hand]

        # Must follow suit if possible
        valid_cards = [card for card in self.hand if card.suit == lead_suit]
        if valid_cards:
            return [self.get_card_index(card) for card in valid_cards]
        else:
            return [self.get_card_index(card) for card in self.hand]

    def get_card_index(self, card):
        return (card.suit_value - 1) * 13 + (card.rank_value - 2)

    def get_card_from_index(self, index):
        suit_idx = index // 13
        rank_idx = index % 13
        suit = list(Card.SUIT_VALUES.keys())[suit_idx]
        rank = list(Card.RANK_VALUES.keys())[rank_idx]
        for card in self.hand:
            if card.suit == suit and card.rank == rank:
                return card
        return None

    def play_card(self, state, lead_suit):
        if self.is_agent:
            valid_actions = self.get_valid_actions(lead_suit)
            action = self.agent.act(state, valid_actions)
            card_to_play = None

            for card in self.hand:
                if self.get_card_index(card) == action:
                    card_to_play = card
                    break

            self.hand.remove(card_to_play)
            return card_to_play, action
        else:
            # Non-agent player - simple strategy
            if lead_suit:
                valid_cards = [card for card in self.hand if card.suit == lead_suit]
                if valid_cards:
                    card_to_play = max(valid_cards)  # Play highest card of lead suit
                else:
                    card_to_play = min(self.hand)  # Play lowest card if can't follow suit
            else:
                card_to_play = max(self.hand)  # Lead with highest card

            self.hand.remove(card_to_play)
            action = self.get_card_index(card_to_play)
            return card_to_play, action


class CardGame:
    def __init__(self, agent_player_id=0):
        self.players = [Player(i, is_agent=(i == agent_player_id)) for i in range(4)]
        self.agent_player = self.players[agent_player_id]
        self.score_array = [0] * 4
        self.rounds_played = 0

    def reset(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.score_array = [0] * 4

        # Reset player hands
        for player in self.players:
            player.hand = self.deck.deal(13)

        # Setup game state tracking
        self.cards_array = [0] * 52
        self.round_array = [0] * 52
        self.hand_arrays = [[0] * 52 for _ in range(4)]
        self.lead_array = [0] * 4
        self.trump_suit = random.randint(0, 3)  # 0-3 representing the suits
        self.trump_array = [0] * 4
        self.trump_array[self.trump_suit] = 1
        self.player_array = [0] * 4

        # Initialize hand arrays
        for player in self.players:
            for card in player.hand:
                card_position = (card.suit_value - 1) * 13 + (card.rank_value - 2)
                self.hand_arrays[player.player_id][card_position] = 1

        # Return initial state
        return self.get_state()

    def get_state(self):
        # Combine all state information into a single array
        state = np.concatenate([
            self.cards_array,
            self.round_array,
            *self.hand_arrays,
            self.lead_array,
            self.trump_array,
            self.player_array
        ])
        return state

    def get_valid_actions(self, player_id, lead_suit):
        return self.players[player_id].get_valid_actions(lead_suit)

    def step(self, lead_player):
        # Play a single round (trick)
        played_cards = []
        actions = []
        lead_suit = None
        self.round_array = [0] * 52
        self.lead_array = [0] * 4

        states = []
        next_states = []

        for i in range(4):
            current_player_id = (lead_player + i) % 4
            current_player = self.players[current_player_id]

            # Get the current state before the player acts
            current_state = self.get_state()

            # First player leads
            if i == 0:
                played_card, action = current_player.play_card(current_state, None)
                lead_suit = played_card.suit
                self.lead_array[Card.SUIT_VALUES[lead_suit] - 1] = 1
            else:
                played_card, action = current_player.play_card(current_state, lead_suit)

            played_cards.append(played_card)
            actions.append(action)

            # Update state
            card_position = (played_card.suit_value - 1) * 13 + (played_card.rank_value - 2)
            self.hand_arrays[current_player_id][card_position] = 0
            self.round_array[card_position] = 1
            self.cards_array[card_position] = 1

            # Save state for the agent if needed
            if current_player.is_agent:
                states.append(current_state)
                next_states.append(self.get_state())

        # Determine winner
        winner_index = self.determine_winner(played_cards, lead_suit, Card.SUITS[self.trump_suit])
        winner_id = (lead_player + winner_index) % 4
        self.score_array[winner_id] += 1

        # Assign rewards - only the agent gets rewards in its memory
        if self.agent_player.player_id == winner_id:
            reward = 1.0
        else:
            reward = 0.0

        # Remember experience for the agent if it played in this round
        for i, (current_player_id, state, action) in enumerate(zip(
                [(lead_player + i) % 4 for i in range(4)],
                states, actions)):

            if self.players[current_player_id].is_agent:
                # The done flag is True if this was the last round
                done = (self.rounds_played == 12)  # 13 rounds total, 0-indexed
                self.agent_player.agent.remember(state, action, reward, next_states[i], done)

        self.rounds_played += 1
        return winner_id

    def determine_winner(self, played_cards, lead_suit, trump_suit):
        trump_cards = [i for i, card in enumerate(played_cards) if card.suit == trump_suit]
        if trump_cards:  # If any trump cards were played
            highest_trump = max(trump_cards, key=lambda i: played_cards[i].rank_value)
            return highest_trump

        # If no trump, highest card of lead suit wins
        lead_cards = [i for i, card in enumerate(played_cards) if card.suit == lead_suit]
        highest_lead = max(lead_cards, key=lambda i: played_cards[i].rank_value)
        return highest_lead

    def play_game(self, train_agent=True, batch_size=64):
        state = self.reset()
        self.rounds_played = 0
        lead_player = 0

        for round_num in range(13):
            winner_id = self.step(lead_player)
            lead_player = winner_id

            # Train the agent after each round if requested
            if train_agent and round_num > 0:
                self.agent_player.agent.train(batch_size=batch_size)

        # Update target network after the game
        self.agent_player.agent.update_target_model()

        # Return the agent's final score
        return self.score_array[self.agent_player.player_id]

    def print_game_stats(self, game_number, show_details=False):
        if show_details:
            print(f"Game {game_number}:")
            print(f"Trump suit: {Card.SUITS[self.trump_suit]}")
            print(f"Scores: {self.score_array}")
            print(f"Agent (Player {self.agent_player.player_id + 1}) score: {self.score_array[self.agent_player.player_id]}/13")
            print(f"Epsilon: {self.agent_player.agent.epsilon:.4f}")
            print("-" * 40)
        else:
            if game_number % 100 == 0:
                print(f"Game {game_number}: Agent score={self.score_array[self.agent_player.player_id]}/13, Epsilon={self.agent_player.agent.epsilon:.4f}")


# Training the DQN agent
def train_dqn_agent(num_games=10000, save_interval=1000):
    game = CardGame(agent_player_id=0)
    scores = []

    for game_num in range(1, num_games + 1):
        score = game.play_game(train_agent=True)
        scores.append(score)
        game.print_game_stats(game_num, show_details=(game_num % 1000 == 0))

        # Save model periodically
        if game_num % save_interval == 0:
            torch.save(game.agent_player.agent.model.state_dict(), f"card_agent_model_{game_num}.pt")

            # Calculate and print average score over last 100 games
            avg_score = sum(scores[-100:]) / min(len(scores), 100)
            print(f"Average score over last 100 games: {avg_score:.2f}")

    # Save final model
    torch.save(game.agent_player.agent.model.state_dict(), "card_agent_model_final.pt")

    # Return the trained agent's model and the score history
    return game.agent_player.agent.model, scores


if __name__ == "__main__":
    print("Starting DQN training for the card game...")
    model, score_history = train_dqn_agent(num_games=10000)

    # You could plot the learning progress here
    print("Training complete!")
