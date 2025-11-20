# Trump System Documentation

## Overview

The game uses a full 52-card deck (13 ranks × 4 suits) with **Spades locked as the trump suit**. Trump cards have special power in winning tricks.

## Trump Rules

### What is Trump?

**Spades** is always the trump suit in this implementation. Trump cards beat all non-trump cards regardless of rank.

### Deck Composition

- **Total cards**: 52 (full standard deck)
- **Suits**: 4 (Clubs, Diamonds, Hearts, Spades)
- **Ranks per suit**: 13 (2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A)
- **Trump suit**: Spades (locked, cannot be changed)
- **Cards per player**: 13 (52 ÷ 4 players)

### Card Values

| Suit | Suit Value | Description |
|------|------------|-------------|
| Clubs | 0 | Regular suit |
| Diamonds | 1 | Regular suit |
| Hearts | 2 | Regular suit |
| **Spades** | **3** | **Trump suit** |

| Rank | Value |
|------|-------|
| 2 | 2 |
| 3-10 | 3-10 |
| Jack (J) | 11 |
| Queen (Q) | 12 |
| King (K) | 13 |
| Ace (A) | 14 |

## Trick-Taking Rules

When evaluating who wins a trick (after 4 cards are played):

### Rule 1: Trump Beats Everything
If any trump card (Spades) is played:
- **The highest trump card wins** the trick
- Non-trump cards cannot win, even if they are higher rank
- Example: **2 of Spades** (trump) beats **Ace of Hearts** (non-trump)

### Rule 2: Multiple Trumps
If multiple trump cards are played:
- **The highest-ranking trump wins**
- Example: **King of Spades** beats **5 of Spades**

### Rule 3: No Trump Played
If no trump cards are played:
- **The highest card in the led suit wins**
- Led suit = the suit of the first card played in the trick
- Off-suit cards (different from led suit) cannot win
- Example: If Hearts is led, **Ace of Hearts** beats **3 of Hearts**, but **King of Diamonds** cannot win

### Rule 4: Following Suit
Players must follow the led suit if they can:
- If a player has cards in the led suit, they must play one
- Only when unable to follow suit can a player:
  - Play a trump card (to try to win)
  - Discard a card from another suit (which cannot win)

## Examples

### Example 1: Trump Dominance
```
Player 0 leads: Ace of Hearts (highest Hearts)
Player 1 plays: King of Hearts 
Player 2 plays: 2 of Spades (TRUMP)
Player 3 plays: 5 of Hearts

Winner: Player 2 (2 of Spades)
```
Even though it's the lowest trump, it beats all non-trump cards.

### Example 2: Trump vs Trump
```
Player 0 leads: 5 of Spades (trump)
Player 1 plays: King of Spades (trump)
Player 2 plays: 7 of Spades (trump)
Player 3 plays: Ace of Hearts (non-trump)

Winner: Player 1 (King of Spades)
```
Highest trump wins when multiple trumps are played.

### Example 3: Normal Trick (No Trump)
```
Player 0 leads: 5 of Hearts
Player 1 plays: Ace of Hearts
Player 2 plays: 7 of Hearts
Player 3 plays: King of Hearts

Winner: Player 1 (Ace of Hearts)
```
When no trump is played, highest card in led suit wins.

### Example 4: Off-Suit Cards
```
Player 0 leads: 2 of Hearts
Player 1 plays: Ace of Diamonds (off-suit, can't follow)
Player 2 plays: 3 of Hearts (led suit)
Player 3 plays: King of Clubs (off-suit, can't follow)

Winner: Player 2 (3 of Hearts)
```
Only led suit or trump can win. Off-suit cards are powerless.

## Implementation Details

### Card Position Mapping

With 52 cards, each card maps to a position 0-51 in the state arrays:

```python
position = suit_value * 13 + (rank_value - 2)
```

| Suit | Position Range | Example |
|------|----------------|---------|
| Clubs (0) | 0-12 | 2♣=0, A♣=12 |
| Diamonds (1) | 13-25 | 2♦=13, A♦=25 |
| Hearts (2) | 26-38 | 2♥=26, A♥=38 |
| Spades/Trump (3) | 39-51 | 2♠=39, A♠=51 |

### Trump Detection

```python
card.is_trump()  # Returns True if card.suit == 'Spades'
```

### Trick Evaluation

Located in `whist.py`, method `_evaluate_trick_winner()`:

1. Identifies the led suit (first card played)
2. Separates trump cards from non-trump cards
3. If any trump cards: returns highest trump
4. If no trump cards: returns highest card in led suit

## State Representation

The game state now uses:
- **ARRAY_LENGTH = 52** (increased from 13)
- Each state array represents all 52 cards
- Neural network input size adjusted: `(52 * 7) + 4 + 4 = 372`

### State Components

1. `cards_array[52]` - Cards played so far
2. `round_array[52]` - Cards played in current trick
3. `hand_array[52]` - Current player's hand
4. `player_array[4]` - Player turn indicator
5. `player1_cards[52]` - Player 1's possible cards
6. `player2_cards[52]` - Player 2's possible cards
7. `player3_cards[52]` - Player 3's possible cards
8. `player4_cards[52]` - Player 4's possible cards
9. `score_array[4]` - Player scores

## Testing

Run the trump system test:

```bash
python test_trump_system.py
```

Tests verify:
1. Trump card identification
2. Full 52-card deck composition
3. Trump beats non-trump
4. Highest trump wins among multiple trumps
5. Led suit rules when no trump played
6. Off-suit cards cannot win

## Strategic Implications

### For Agents
- **Trump management**: When to play trump vs. save for later
- **Trump counting**: Track which trumps have been played
- **Lead strategy**: When to lead trump vs. other suits
- **Defensive play**: When to use trump to prevent opponent from winning

### For Training
- Agents must learn that trump cards are powerful
- Partner coordination: Don't trump partner's winning non-trump
- Suit management: Balance trump usage across the game
- Counting: Track remaining trumps to assess risk

## Related Files

- `whist_game.py` - Card and Deck classes with trump implementation
- `whist.py` - Whist game logic with trump-aware trick evaluation
- `test_trump_system.py` - Comprehensive trump system tests
- `model_training.py` - Training with 52-card state representation
