# Game Logging System

## Overview

The game logging system automatically saves detailed game logs every 50 games to .txt files in the `game_logs/` directory.

## Features

- **Automatic logging**: Every 50th game is logged (games 50, 100, 150, etc.)
- **Starting player tracking**: Shows which player (North/East/South/West) starts the game
- **Complete hand information**: All players' starting hands grouped by suit
- **Trick-by-trick recording**: Every card played in every trick
- **Final results**: Scores and winner information

## Configuration

In `model_training.py`:
```python
LOG_GAME_EVERY = 50  # Save detailed game logs every 50 games
```

Change this value to log more or fewer games.

## File Format

Files are saved as `game_XXXX.txt` where XXXX is the zero-padded game number.

Example: `game_0050.txt`, `game_0100.txt`, `game_0150.txt`

## Sample Log

```
=== Game 50 ===
Date: 2025-11-20 11:55:46
Starting player: East

North hand: Clubs: 5 8 Q | Diamonds: 5 6 7 | Hearts: 5 6 9 K | Spades: 2 6 10
East hand: Clubs: 3 6 A | Diamonds: 3 K A | Hearts: 7 10 A | Spades: 5 7 9 J
South hand: Clubs: 10 J K | Diamonds: 2 10 J Q | Hearts: 2 3 4 J Q | Spades: K
West hand: Clubs: 2 4 7 9 | Diamonds: 4 8 9 | Hearts: 8 | Spades: 3 4 8 Q A

Trick 1:
  E: 3 of Clubs [Strategy]
  S: 10 of Clubs [Agent: Q=0.800]
  W: 2 of Clubs [Strategy]
  N: 5 of Clubs [Agent: Q=0.800]

Trick 2:
  E: 6 of Clubs [Strategy]
  S: J of Clubs [Random exploration]
  W: 4 of Clubs [Strategy]
  N: 8 of Clubs [Random exploration]

Trick 3:
  E: A of Clubs [Strategy]
  S: K of Clubs [Agent: Q=0.900]
  W: 7 of Clubs [Strategy]
  N: Q of Clubs [Agent: Q=0.900]

Final Scores:
  North: 1 tricks
  East: 2 tricks
  South: 0 tricks
  West: 0 tricks

Winner: Team 2 (East-West) - 2 vs 1
```

## Decision Information

Each card played now includes decision information:

- **[Agent: Q=X.XXX]** - Agent made a deliberate decision based on Q-values. Higher Q-values indicate more confidence (typically ranges from -10 to +10).
- **[Random exploration]** - Agent chose a random action for exploration (epsilon-greedy strategy).
- **[Strategy]** - Rule-based strategic player (East/West positions) made the decision.

This allows you to:
- Track when agents are exploring vs. exploiting learned knowledge
- See confidence levels in agent decisions
- Understand how decision-making evolves during training
- Identify patterns in strategic vs. random play

## Usage

The logging system is automatically integrated into the training process. Simply run:

```bash
python model_training.py
```

Logs will be saved to the `game_logs/` directory as the training progresses.

## Testing

Test the game logger:

```bash
python test_game_logger.py
```

This creates a sample game log and verifies the format is correct.

## Implementation Details

### GameLogger Class

Located in `game_logger.py`, provides:
- `start_game()` - Initialize logging for a new game
- `log_card_played(player_idx, card, decision_type, certainty)` - Record each card played with decision info
- `end_game()` - Finalize and save the log
- `_format_hand()` - Format hands grouped by suit

#### Decision Types

- **'agent'** - DQN agent made a decision based on Q-values (with certainty value)
- **'random'** - Random exploration move (epsilon-greedy)
- **'strategy'** - Rule-based strategic decision (East/West players)

#### Certainty Values

For agent decisions, the certainty is the Q-value associated with the chosen action:
- Higher values indicate stronger confidence in the action
- Values typically range from -10 to +10
- Early in training, values may be close to 0 or negative
- As training progresses, values should increase for good moves

### Position Names

- Position 0: North (N) - DQN Agent
- Position 1: East (E) - Strategic player
- Position 2: South (S) - DQN Agent
- Position 3: West (W) - Strategic player

### Hand Format

Hands are displayed grouped by suit with ranks in order:
```
Suit: rank1 rank2 rank3 | Suit: rank1 rank2 | ...
```

Example: `Clubs: 3 6 K | Diamonds: 2 6 A | Hearts: 8 | Spades: 2 3 5 J Q A`

## Benefits

1. **Training Analysis**: Review how agents play at different stages of training
2. **Decision Transparency**: See when agents explore vs. exploit learned knowledge
3. **Confidence Tracking**: Monitor Q-value evolution as training progresses
4. **Debugging**: Identify issues with game logic or agent decisions
5. **Strategy Understanding**: See patterns in how agents and strategic players make moves
6. **Documentation**: Permanent record of interesting games

## Decision Pattern Analysis

By examining the logs, you can:
- **Track exploration rate**: Count `[Random exploration]` vs `[Agent: Q=...]` occurrences
- **Monitor confidence growth**: Watch Q-values increase as agents learn
- **Compare strategies**: See how agent decisions differ from rule-based play
- **Identify learning milestones**: Notice when agents start making consistently high-Q decisions

## Notes

- The `game_logs/` directory is in `.gitignore` to avoid committing large numbers of log files
- Each log file is approximately 2-3 KB depending on the number of tricks
- For 1000 games with logging every 50 games, expect ~20 log files (~50 KB total)
