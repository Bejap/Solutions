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
Date: 2025-11-20 11:41:09
Starting player: West

North hand: Clubs: 3 6 K | Diamonds: 2 6 A | Hearts: 8 | Spades: 2 3 5 J Q A
East hand: Clubs: 5 7 10 Q A | Diamonds: 3 7 8 10 | Hearts: 3 | Spades: 4 6 10
South hand: Clubs: 4 8 9 | Diamonds: 4 K | Hearts: 4 6 J Q A | Spades: 7 8 9
West hand: Clubs: 2 J | Diamonds: 5 9 J Q | Hearts: 2 5 7 9 10 K | Spades: K

Trick 1:
  W: 2 of Clubs
  N: 3 of Clubs
  E: 5 of Clubs
  S: 4 of Clubs

Trick 2:
  W: J of Clubs
  N: 6 of Clubs
  E: 7 of Clubs
  S: 8 of Clubs

Trick 3:
  W: 5 of Diamonds
  N: K of Clubs
  E: 10 of Clubs
  S: 9 of Clubs

Final Scores:
  North: 2 tricks
  East: 0 tricks
  South: 1 tricks
  West: 0 tricks

Winner: Team 1 (North-South) - 3 vs 0
```

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
- `log_card_played()` - Record each card played
- `end_game()` - Finalize and save the log
- `_format_hand()` - Format hands grouped by suit

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
2. **Debugging**: Identify issues with game logic or agent decisions
3. **Strategy Understanding**: See patterns in how agents and strategic players make moves
4. **Documentation**: Permanent record of interesting games

## Notes

- The `game_logs/` directory is in `.gitignore` to avoid committing large numbers of log files
- Each log file is approximately 2-3 KB depending on the number of tricks
- For 1000 games with logging every 50 games, expect ~20 log files (~50 KB total)
