# Inheritance Structure Documentation

## Overview

This document describes the comprehensive inheritance hierarchy implemented in the Whist DQN project. The refactored codebase uses Object-Oriented Programming (OOP) principles to create a maintainable, extensible, and type-safe architecture.

## Architecture Diagram

```
base_classes.py (Abstract Base Classes)
│
├── BaseCard
│   └── Card (whist_game.py)
│
├── BaseDeck  
│   └── Deck (whist_game.py)
│
├── BasePlayer
│   └── Player (whist_game.py)
│
├── BaseGame
│   └── Whist (whist.py)
│
├── BaseAgent
│   └── DQNAgent (simple_whist_DQN.py)
│
└── BaseStrategy
    └── EWStrategy (ew_strategy.py)
```

## Base Classes (base_classes.py)

### BaseAgent
Abstract base class for all agents (learning and rule-based).

**Required Methods:**
- `choose_action(state, valid_actions, epsilon)` - Choose an action given a state
- `update(transition)` - Update the agent based on a state transition

**Implementations:**
- `DQNAgent` - Deep Q-Network learning agent

### BasePlayer
Abstract base class for player representations in the game.

**Required Methods:**
- `action(choice)` - Execute a card play action
- `get_hand()` - Get the player's current hand

**Common Attributes:**
- `id` - Player identifier (1-4)
- `hand` - Current cards in hand
- `tricks_won` - Number of tricks won
- `known_actions` - Observed actions from other players

**Implementations:**
- `Player` - Standard whist player

### BaseGame
Abstract base class for card game implementations.

**Required Methods:**
- `reset(seed)` - Reset the game state
- `step(action)` - Execute one game step
- `get_valid_actions(player)` - Get valid actions for a player
- `deal_cards()` - Deal cards to all players

**Implementations:**
- `Whist` - Full whist game with trump system and team play

### BaseStrategy
Abstract base class for playing strategies.

**Required Methods:**
- `choose_action(player, valid_actions)` - Choose an action based on strategy rules

**Implementations:**
- `EWStrategy` - Bridge-like strategic play (80% strategic, 20% random)

### BaseCard
Abstract base class for playing cards.

**Required Methods:**
- `__repr__()` - String representation
- `compare_to(other, led_suit)` - Compare cards for trick evaluation

**Implementations:**
- `Card` - Standard playing card with rank and suit

### BaseDeck
Abstract base class for card decks.

**Required Methods:**
- `shuffle()` - Shuffle the deck
- `deal(num_cards)` - Deal cards from the deck

**Implementations:**
- `Deck` - 52-card standard deck (4 suits × 13 ranks)

## Concrete Classes

### Card (extends BaseCard)
Represents a single playing card.

**Attributes:**
- `suit` - Card suit (Clubs, Diamonds, Hearts, Spades)
- `rank` - Card rank (2-10, J, Q, K, A)
- `rank_value` - Numeric value of rank (2-14)
- `suit_value` - Numeric value of suit (0-3)

**Special Methods:**
- `is_trump()` - Check if card is trump (Spades)
- `compare_to(other, led_suit)` - Advanced comparison considering trump and led suit

**Trump Rules:**
1. Trump (Spades) beats any non-trump card
2. Highest trump wins if multiple trumps played
3. If no trump, highest card in led suit wins

### Deck (extends BaseDeck)
Represents a full 52-card deck.

**Features:**
- Automatically creates all 52 cards (4 suits × 13 ranks)
- Supports shuffling and dealing
- Pop-based dealing (modifies deck)

### Player (extends BasePlayer)
Represents a player in the whist game.

**Key Methods:**
- `action(choice)` - Play the card at index `choice` in sorted hand
- `observe(player_id, action)` - Record another player's action
- `return_other_hand(hand, list_length)` - Track known cards for all players
- `_sort_hand()` - Sort hand by suit and rank

**Hand Management:**
- Hand is automatically sorted before each action
- Tracking system maintains knowledge of other players' possible cards

### Whist (extends BaseGame)
Main game engine implementing whist rules.

**Key Features:**
- Full 52-card deck with Spades as locked trump
- 4 players in 2 teams (0-2 vs 1-3)
- Follow suit rules enforced
- Trump system with trick evaluation
- Reward system for DQN training

**State Representation:**
The game provides a structured state with 9 components:
1. `cards_array` - All cards played so far (52 elements)
2. `round_array` - Cards played in current trick (52 elements)
3. `hand_array` - Current player's hand (52 elements)
4. `player_array` - Current player indicator (4 elements)
5. `player1_cards` - Player 1's known/possible cards (52 elements)
6. `player2_cards` - Player 2's known/possible cards (52 elements)
7. `player3_cards` - Player 3's known/possible cards (52 elements)
8. `player4_cards` - Player 4's known/possible cards (52 elements)
9. `score_array` - Tricks won by each player (4 elements)

**Total state size:** (52 × 7) + 4 + 4 = 372 features

### DQNAgent (extends BaseAgent)
Deep Q-Network reinforcement learning agent.

**Architecture:**
- Multi-input neural network
- 3 hidden layers (128, 64, 32 units)
- Dropout regularization (35%)
- Experience replay memory
- Target network for stability

**Key Methods:**
- `choose_action(state, valid_actions, epsilon)` - Epsilon-greedy action selection
- `update(transition)` - Update replay memory
- `train(terminal_state, step)` - Train on minibatch
- `get_qs(state)` - Get Q-values for a state

**Training Features:**
- Experience replay with 100K memory
- Target network updated every 5 terminal states
- Batch size of 32 for stable learning
- Support for different gamma values per agent

### EWStrategy (extends BaseStrategy)
Rule-based strategic player for East-West positions.

**Strategy Rules:**
- **Leading:** Fourth-best from longest suit or top of sequence
- **Following:** Win with lowest winning card or conserve high cards
- **Partner Awareness:** Don't compete when partner is winning
- **Random Play:** 20% of time for unpredictability

**Key Methods:**
- `choose_action(player, valid_actions)` - Select action based on strategy
- `_strategic_action(player, valid_actions)` - Apply strategic rules
- `_choose_lead(valid_cards)` - Select card to lead
- `_choose_follow(player, valid_cards)` - Select card to follow

## Benefits of Inheritance Structure

### 1. Polymorphism
All agents, strategies, and players follow common interfaces, allowing:
- Uniform treatment in game loops
- Easy swapping of agent types
- Consistent API across implementations

### 2. Extensibility
New components can be added easily:
```python
# Add a new agent type
class RandomAgent(BaseAgent):
    def choose_action(self, state, valid_actions, epsilon=None):
        return random.choice(valid_actions)
    
    def update(self, transition):
        pass  # Random agent doesn't learn

# Add a new strategy
class AggressiveStrategy(BaseStrategy):
    def choose_action(self, player, valid_actions):
        # Always play highest card
        player._sort_hand()
        return valid_actions[-1]
```

### 3. Type Safety
Abstract base classes ensure:
- Required methods are implemented
- Consistent method signatures
- Clear contracts between components

### 4. Code Reuse
Common functionality in base classes:
- `BasePlayer` handles observations and basic actions
- `BaseDeck` provides common deck operations
- `BaseAgent` defines agent interface

### 5. Maintainability
- Clear separation of concerns
- Well-defined responsibilities
- Easy to understand and modify

## Constants Configuration

All configuration is centralized in `constants.py`:

```python
# Game configuration
ARRAY_LENGTH = 52  # Full 52-card deck
NUM_PLAYERS = 4
DQN_AGENT_POSITIONS = [0, 2]  # North and South

# Agent types
AGENT_TYPE_DQN = "DQN"
AGENT_TYPE_STRATEGY = "STRATEGY"
STRATEGY_EW = "EW_STRATEGY"

# Training parameters
DEFAULT_GAMMA_VALUES = [0.99, 0.95, 0.90, 0.85]
REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 32

# Network architecture
HIDDEN_LAYER_1_SIZE = 128
HIDDEN_LAYER_2_SIZE = 64
HIDDEN_LAYER_3_SIZE = 32
DROPOUT_RATE = 0.35
```

## Usage Example

```python
from whist import Whist
from simple_whist_DQN import DQNAgent
from ew_strategy import EWStrategy
from constants import ARRAY_LENGTH, NUM_PLAYERS

# Create game
game = Whist([1, 2, 3, 4])

# Create agents (polymorphic - both follow BaseAgent interface)
agent_north = DQNAgent((ARRAY_LENGTH * 7) + 4 + 4, gamma=0.99, agent_id=0)
agent_south = DQNAgent((ARRAY_LENGTH * 7) + 4 + 4, gamma=0.99, agent_id=2)

# Create strategies
strategy_east = EWStrategy(2, game)
strategy_west = EWStrategy(4, game)

# All agents can be used uniformly
agents = [agent_north, None, agent_south, None]
strategies = {1: strategy_east, 3: strategy_west}

# Game loop
state = game.reset()
for player_idx in range(NUM_PLAYERS):
    player = game.players[player_idx]
    valid_actions = game.get_valid_actions(player)
    
    if agents[player_idx]:
        # Use learning agent
        action = agents[player_idx].choose_action(state, valid_actions, epsilon=0.1)
    else:
        # Use strategy
        action = strategies[player_idx].choose_action(player, valid_actions)
    
    new_state, reward, done = game.step(action)
```

## Testing

Run the demonstration to see the inheritance structure in action:

```bash
python demo_inheritance.py
```

This will show:
- All inheritance relationships
- Polymorphic behavior
- Type checking with isinstance()
- Method availability
- Benefits summary

## Future Extensions

The inheritance structure makes it easy to add:

1. **New Agent Types:**
   - Monte Carlo Tree Search agent
   - Human player agent
   - Hybrid agents

2. **New Strategies:**
   - Aggressive play strategy
   - Defensive play strategy
   - Adaptive strategies

3. **New Game Variants:**
   - Different trump systems
   - Different team configurations
   - Variant rule sets

4. **Analysis Tools:**
   - Game state analyzers
   - Strategy comparers
   - Performance trackers

All extensions just need to inherit from the appropriate base class and implement the required methods!
