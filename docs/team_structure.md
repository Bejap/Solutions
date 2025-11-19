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
  - These positions play randomly
  - Used as opponents for agent training

## Game Mechanics

- The game is played with 4 players in teams of 2
- North and South (Team 1) compete against East and West (Team 2)
- Each player plays one card per trick
- The highest card wins the trick
- The team that wins the most tricks wins the game

## Training Configuration

The training scripts (`model_training.py`) are configured to:
- Train agents only for North (position 0) and South (position 2)
- Use random actions for East (position 1) and West (position 3)
- This allows the agents to learn cooperative strategies while playing against random opponents
