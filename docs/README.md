# Documentation

This folder contains documentation for the Deep Simple Whist DQN implementation.

## Contents

### Core Documentation
- **[architecture.md](architecture.md)** - Detailed neural network architecture documentation
- **[INHERITANCE_STRUCTURE.md](INHERITANCE_STRUCTURE.md)** - OOP inheritance and class hierarchy
- **[CARD_EMBEDDING_GUIDE.md](CARD_EMBEDDING_GUIDE.md)** - Card embedding system guide
- **[CONSTANTS_GUIDE.md](CONSTANTS_GUIDE.md)** - Configuration constants reference

### Training & Features
- **[EMBEDDED_TRAINING_GUIDE.md](EMBEDDED_TRAINING_GUIDE.md)** - Guide to embedded agent training
- **[GPU_NPU_GUIDE.md](GPU_NPU_GUIDE.md)** - GPU/NPU acceleration setup
- **[FEATURE_IMPLEMENTATION_GUIDE.md](FEATURE_IMPLEMENTATION_GUIDE.md)** - Feature implementation guide
- **[optimization_suggestions.md](optimization_suggestions.md)** - Training optimization tips

### Game Rules & Structure
- **[team_structure.md](team_structure.md)** - Player positions and team assignments
- **[trump_system.md](trump_system.md)** - Trump rules and card comparison
- **[reward_system.md](reward_system.md)** - Reward structure and calculation

### Performance & Analysis
- **[performance_improvements.md](performance_improvements.md)** - Performance optimization history
- **[BEFORE_AFTER_COMPARISON.md](BEFORE_AFTER_COMPARISON.md)** - Comparison before/after refactoring
- **[benchmark_games.md](benchmark_games.md)** - Game benchmarking results
- **[game_logging.md](game_logging.md)** - Game logging system documentation

## Quick Start

1. Read [INHERITANCE_STRUCTURE.md](INHERITANCE_STRUCTURE.md) to understand the architecture
2. Read [CARD_EMBEDDING_GUIDE.md](CARD_EMBEDDING_GUIDE.md) for the embedding system
3. Read [EMBEDDED_TRAINING_GUIDE.md](EMBEDDED_TRAINING_GUIDE.md) to start training
4. Review [GPU_NPU_GUIDE.md](GPU_NPU_GUIDE.md) for hardware acceleration
5. Check [CONSTANTS_GUIDE.md](CONSTANTS_GUIDE.md) for configuration options

## Recent Updates

### Prioritized Experience Replay
- Implemented SumTree-based prioritized replay for efficient learning
- TD-error based sampling with importance weights
- Configurable via constants in `Whist/utils/constants.py`

### Organized Training Structure
- Training files reorganized by approach (classic, embedded, common)
- Model storage organized by type (classic, embedded, dueling)
- Timestamped model filenames with average reward tracking

### Advanced Features
- Dueling DQN architecture support
- N-step returns for multi-step learning
- Learning rate scheduling
- Epsilon scheduling strategies

## Key Points

- North and South (positions 0 and 2) are DQN agents on Team 1
- East and West (positions 1 and 3) use strategic rule-based play on Team 2
- Spades is the locked trump suit
- Models are saved with timestamps and performance metrics
- Prioritized replay is enabled by default for faster learning
