# Game Size Configuration Guide

## Overview

The Whist DQN game now supports flexible game sizes! You can easily adjust the number of cards dealt to each player by changing a single constant.

## Quick Start

To change the game size, edit the `CARDS_PER_PLAYER` constant in `constants.py`:

```python
# In constants.py
CARDS_PER_PLAYER = 11  # Change this value
```

**That's it!** All other values (ARRAY_LENGTH, STATE_SIZE, ACTION_SIZE, etc.) are automatically calculated.

## Supported Game Sizes

| Cards Per Player | Total Cards | Game Type | STATE_SIZE |
|-----------------|-------------|-----------|------------|
| 9 | 36 | Small game | 260 |
| 11 | 44 | Medium game | 316 |
| 13 | 52 | Full Whist | 372 |

## What Gets Calculated Automatically

When you change `CARDS_PER_PLAYER`, the following values are automatically updated:

1. **ARRAY_LENGTH** = CARDS_PER_PLAYER × NUM_PLAYERS
   - Example: 11 cards × 4 players = 44 total cards

2. **STATE_SIZE** = (ARRAY_LENGTH × 7) + 8
   - Example: (44 × 7) + 8 = 316

3. **ACTION_SIZE** = CARDS_PER_PLAYER
   - Example: 11 possible actions (one for each card in hand)

4. **Input Dimensions**:
   - GAME_INPUT_SIZE = ARRAY_LENGTH × 2
   - PLAYER_INPUT_SIZE = ARRAY_LENGTH + 4
   - TRACKING_INPUT_SIZE = ARRAY_LENGTH × 4
   - SCORE_INPUT_SIZE = 4 (always 4 players)

## Example: Changing to 11 Cards Per Player

### Step 1: Edit constants.py

```python
# constants.py
CARDS_PER_PLAYER = 11  # Changed from 13 to 11
```

### Step 2: Run Your Training

```bash
python model_training.py
```

That's it! The neural network will automatically:
- Create the correct input/output dimensions (316 input, 11 output)
- Deal 11 cards to each player
- Track 44 total cards in the state arrays
- Adjust all game logic accordingly

## Verification

You can verify the configuration is correct:

```python
from constants import CARDS_PER_PLAYER, ARRAY_LENGTH, STATE_SIZE, ACTION_SIZE

print(f"Cards per player: {CARDS_PER_PLAYER}")
print(f"Total cards: {ARRAY_LENGTH}")
print(f"State size: {STATE_SIZE}")
print(f"Action size: {ACTION_SIZE}")
```

## Network Architecture for Smaller Games

For smaller games (9-11 cards), you may want to use a smaller neural network for faster training:

```python
# constants.py - Optional network size adjustment for smaller games
HIDDEN_LAYER_1_SIZE = 64   # Default: 128
HIDDEN_LAYER_2_SIZE = 32   # Default: 64
HIDDEN_LAYER_3_SIZE = 16   # Default: 32
```

These are optional - the default sizes work fine for all game sizes.

## Important Notes

1. **Models are NOT compatible across game sizes**
   - A model trained with 13 cards cannot be used with 11 cards
   - You must train separate models for each game size

2. **Deck must have enough cards**
   - The deck is always created with 52 cards (full deck)
   - Make sure `CARDS_PER_PLAYER × NUM_PLAYERS ≤ 52`
   - Valid range: 1-13 cards per player (for 4 players)

3. **Training considerations**
   - Smaller games train faster (fewer states to explore)
   - Smaller games may need fewer episodes to converge
   - Consider adjusting `DEFAULT_NUM_GAMES` for smaller games

## Troubleshooting

### Error: "list index out of range" during training

**Cause**: Model was created with wrong STATE_SIZE

**Solution**: 
1. Delete old model weights from `Weights/` and `Models/` directories
2. Make sure you've changed `CARDS_PER_PLAYER` in constants.py
3. Restart training from scratch

### Cards not dealing correctly

**Cause**: Old cached imports

**Solution**: Restart Python interpreter or Jupyter kernel

### Model dimensions mismatch

**Cause**: Trying to load a model trained with different CARDS_PER_PLAYER

**Solution**: Train a new model with the current CARDS_PER_PLAYER setting

## Benefits of This Approach

✅ **Single point of configuration** - Change one value, everything updates
✅ **Automatic calculations** - No manual math required
✅ **Type safe** - All dependencies use the same constants
✅ **Easy experimentation** - Try different game sizes quickly
✅ **Clear and documented** - Comments explain what each value represents

## Example Training Script

```python
from constants import CARDS_PER_PLAYER
from training_logic import WhistTrainer

# The trainer automatically uses CARDS_PER_PLAYER from constants
print(f"Training with {CARDS_PER_PLAYER} cards per player")

trainer = WhistTrainer(
    num_games=500,  # Fewer games for smaller variants
    epsilon=1.0,
    save_every=100
)

trainer.train()
trainer.plot_results()
```

## Migration from Old Code

If you have old code that manually set these values, you can now simplify:

### Before (Old Way - 3-4 values to change):
```python
ARRAY_LENGTH = 44  # Manual calculation
STATE_SIZE = 316   # Manual calculation
ACTION_SIZE = 11   # Manual setting
# Plus changes in deal_cards() method
```

### After (New Way - 1 value to change):
```python
CARDS_PER_PLAYER = 11  # Everything else calculated automatically!
```

---

**Happy training with flexible game sizes!** 🎮
