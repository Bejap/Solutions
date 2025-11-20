# Team Structure

## Player Positions

In this Whist implementation, players are arranged in the traditional cardinal directions:

```
        North (Player 0)
             |
West (3) ----+---- East (1)
             |
        South (Player 2)
```

## Team Assignments

- **Team 1 (North-South Partnership)**: Players 0 and 2
  - These positions are controlled by DQN agents
  - They work together to win tricks

- **Team 2 (East-West Partnership)**: Players 1 and 3
  - These positions use strategic rule-based play (80% of the time)
  - Play randomly 20% of the time for variety
  - Used as opponents for agent training

## East-West Strategy

East-West players follow sophisticated bridge-like rules to provide consistent, challenging opponents:

### Following Suit
- Always follow suit if able (enforced by game rules)
- Play strategically based on trick situation

### Winning Tricks
- Win the trick only if it helps your side
- Heuristics:
  - Win if you have the highest remaining card in the suit
  - Win if it allows you to lead a long suit (>2 cards)
  - Don't win if partner is already winning

### When Not Winning
- Play the lowest card to conserve high cards
- Signals to partner (attitude: high encourages, low discourages)

### Leading
- Lead fourth-best from longest suit (≥4 cards)
- Lead top of sequence (K from KQJ, Q from QJ)
- Lead from strongest suit otherwise

### Randomness
- 20% of plays are random to prevent predictability
- This helps North-South agents learn to respond to varied play styles

## Game Mechanics

- The game is played with 4 players in teams of 2
- North and South (Team 1) compete against East and West (Team 2)
- Each player plays one card per trick
- The highest card wins the trick
- The team that wins the most tricks wins the game

## Training Configuration

The training scripts (`model_training.py`) are configured to:
- Train agents only for North (position 0) and South (position 2)
- Use strategic rule-based play for East (position 1) and West (position 3)
- This allows the agents to learn cooperative strategies while playing against consistent, challenging opponents
