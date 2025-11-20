# Benchmark Games Feature

## Overview

Every 100th game (100, 200, 300, 400, etc.) uses a **fixed random seed** to create identical, reproducible games. This allows tracking whether the model improves on the exact same scenario over time.

## How It Works

### Configuration
- **Benchmark Frequency**: Every 100 games (`BENCHMARK_GAME_EVERY = 100`)
- **Fixed Seed**: 42 (`BENCHMARK_SEED = 42`)

### Implementation
When a benchmark game is detected (episode % 100 == 0):
1. Game is reset with a fixed seed: `game.reset(seed=42)`
2. This produces identical:
   - Card distribution (all players get the same hands)
   - Starting player
   - Deck order
3. Game is marked in logs as `[BENCHMARK - Fixed Seed]`
4. Training log shows: `Episode 100: BENCHMARK GAME (using fixed seed 42)`

## Benefits

### 1. **Track Model Improvement**
Compare performance on the exact same scenario:
- Game 100: Initial performance
- Game 200: After 100 more training games
- Game 300: After 200 more training games
- etc.

### 2. **Consistent Evaluation**
- Removes randomness from evaluation
- Fair comparison across training epochs
- Clear signal of learning progress

### 3. **Debugging**
- Replay exact scenarios that caused issues
- Verify fixes work on specific cases
- Understand decision-making in controlled environment

## Example Usage

### During Training
```python
# Games 1-99: Random training
# Game 100: Fixed seed (benchmark)
# Games 101-199: Random training  
# Game 200: Fixed seed (same as game 100)
# Games 201-299: Random training
# Game 300: Fixed seed (same as games 100, 200)
```

### Log Output
```
=== Game 300 [BENCHMARK - Fixed Seed] ===
Date: 2025-11-20 13:00:00
Trump: Spades
Starting player: South

North hand: Clubs: Q K A | Diamonds: 5 7 9 10 | Hearts: 4 9 10 | Spades: 4 5 J
East hand: Clubs: 2 3 7 | Diamonds: 3 4 6 K | Hearts: 5 8 K A | Spades: 3 Q
South hand: Clubs: 4 5 8 9 10 | Diamonds: 2 8 J | Hearts: J | Spades: 6 10 K A
West hand: Clubs: 6 J | Diamonds: Q A | Hearts: 2 3 6 7 Q | Spades: 2 7 8 9
```

## Analysis Tips

### Compare Benchmark Games
1. Look at decision quality (Q-values) across benchmark games
2. Track whether agents make better plays over time
3. Measure win rate on benchmark scenario
4. Observe exploration vs exploitation balance

### Metrics to Track
- **Rewards**: Total rewards earned in benchmark games
- **Decisions**: Proportion of agent decisions vs random exploration
- **Q-values**: Confidence levels in chosen actions
- **Win Rate**: Whether Team 1 (agents) wins the benchmark game

### Example Analysis
```python
# Pseudocode for tracking benchmark performance
benchmark_results = {
    100: {'rewards': [2.5, 3.0], 'winner': 'Team 2', 'avg_q': 0.5},
    200: {'rewards': [4.0, 5.5], 'winner': 'Team 1', 'avg_q': 1.2},
    300: {'rewards': [5.5, 6.0], 'winner': 'Team 1', 'avg_q': 2.1},
    # Shows clear improvement: higher rewards, wins, confidence
}
```

## Configuration

To change benchmark frequency or seed, edit `model_training.py`:

```python
BENCHMARK_GAME_EVERY = 100  # Change to 50 for more frequent benchmarks
BENCHMARK_SEED = 42         # Change to different seed for different scenario
```

## Technical Details

### Seeding Both RNGs
The implementation seeds both random number generators:
```python
random.seed(seed)      # Python's random module (deck shuffle)
np.random.seed(seed)   # NumPy random (starting player)
```

This ensures complete reproducibility of:
- Card shuffle order
- Starting player selection
- Any other random elements

### Why Seed 42?
The choice of 42 is arbitrary but conventional in machine learning (reference to "The Hitchhiker's Guide to the Galaxy"). Any seed would work - the important part is consistency.

## Testing

Run the benchmark test suite:
```bash
python test_benchmark_games.py
```

This verifies:
- Same seed produces identical games
- Different seeds produce different games
- Multiple resets with same seed work correctly
- Benchmark games across episodes are identical

## Is This Smart?

**Yes!** This is a standard practice in reinforcement learning research:

### Advantages
✓ Objective performance measurement
✓ Removes variance from evaluation
✓ Easier to detect overfitting
✓ Facilitates debugging and analysis
✓ Common in academic papers

### Best Practices
- Keep 90%+ of games random (for diverse training)
- Use multiple benchmark seeds (if needed)
- Track benchmark performance separately from overall performance
- Don't train differently on benchmark games (no special treatment)

### Similar Concepts
- Test sets in supervised learning
- Benchmark suites in game AI (e.g., Atari benchmarks)
- Validation sets in deep learning
- Deterministic evaluation in AlphaGo/AlphaZero papers
