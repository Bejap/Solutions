# Neural Network Architecture

This document describes the Deep Q-Network (DQN) architecture used in the Simple Whist implementation.

## Overview

The DQN agent uses a multi-input neural network architecture designed to process different aspects of the game state separately before combining them for decision-making.

## Network Architecture

### Input Layers

The network takes four separate inputs, each representing different aspects of the game state:

1. **Game Input**: Shape `(26,)` - Cards played in the current round
   - `cards_array` (13 values): Current trick cards
   - `round_array` (13 values): Round information

2. **Player Input**: Shape `(17,)` - Player's hand and position
   - `hand_array` (13 values): Cards in player's hand
   - `player_array` (4 values): Player position/role information

3. **Tracking Input**: Shape `(52,)` - Card tracking across all players
   - Four 13-element arrays tracking cards played by each player
   - Helps agent remember what cards have been played

4. **Score Input**: Shape `(4,)` - Current game score
   - Score information for all four players/teams

### Feature Processing Layers

Each input is processed through a dedicated Dense layer to extract relevant features:

- **Game Features**: Dense(26, activation='relu') 
- **Player Features**: Dense(17, activation='relu')
- **Tracking Features**: Dense(52, activation='relu')
- **Score Features**: Dense(13, activation='relu')

### Combined Processing Layers

After feature extraction, all inputs are concatenated and processed through three hidden layers with dropout for regularization:

1. **Hidden Layer 1**: Dense(128, activation='relu')
   - **Dropout 1**: Dropout(0.35) - Prevents overfitting
   
2. **Hidden Layer 2**: Dense(64, activation='relu')
   - **Dropout 2**: Dropout(0.35) - Regularization
   
3. **Hidden Layer 3**: Dense(32, activation='relu')
   - **Dropout 3**: Dropout(0.35) - Additional regularization

### Output Layer

- **Q-Value Output**: Dense(13, activation='linear')
  - 13 outputs corresponding to Q-values for each possible card action
  - Linear activation for Q-value estimation

## Dropout Implementation

Dropout layers with rate 0.35 are applied after each hidden layer:
- **Purpose**: Prevent overfitting and improve generalization
- **Rate**: 0.35 (35% of neurons randomly dropped during training)
- **Benefit**: Helps the network learn more robust features and generalize better to unseen game states

## Model Compilation

- **Optimizer**: Adam (default learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
  - Appropriate for Q-value regression

## Total Parameters

The network has approximately:
- Input processing: ~7,000 parameters
- Hidden layers: ~17,000 parameters  
- Output layer: ~400 parameters
- **Total**: ~24,000+ trainable parameters

## Design Rationale

### Multi-Input Architecture
- Separates different types of game information
- Allows the network to learn specialized feature representations
- More interpretable than a single concatenated input

### Progressive Dimension Reduction
- 128 → 64 → 32 hidden units
- Gradually compresses information while maintaining important features
- Balanced between model capacity and overfitting risk

### Dropout Regularization
- Added at 0.35 rate based on optimization suggestions
- Applied after each hidden layer for consistent regularization
- Helps prevent memorization of specific game sequences

## Training Strategy

The network uses:
- **Experience Replay**: Stores past experiences in a replay buffer
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Mini-batch Training**: Processes 8 samples at a time (MINIBATCH_SIZE = 8)

## Future Improvements

Potential enhancements mentioned in `optimization_suggestions.md`:
- Try LeakyReLU or ELU activation functions
- Experiment with soft target updates (τ = 0.001)
- Implement prioritized experience replay
- Adjust learning rate with scheduling
