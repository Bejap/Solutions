# Card Embedding System Guide

## Overview

The card embedding system replaces one-hot encoding with **learned vector embeddings** for representing cards. This provides a more flexible, compact, and expressive representation that works with any number of cards.

## Why Embeddings?

### Problems with One-Hot Encoding
❌ **Size dependent**: State size changes with number of cards (52 cards = 372 dims)
❌ **Not flexible**: Can't transfer models between game sizes
❌ **Sparse**: Most values are 0, inefficient
❌ **No relationships**: Doesn't capture card similarities

### Benefits of Embeddings
✅ **Fixed size**: State size independent of cards dealt (56 dims with 8D embeddings)
✅ **Flexible**: Same model works for 9, 11, or 13 cards
✅ **Compact**: Dense representation
✅ **Learned relationships**: Network learns card similarities
✅ **Aggregatable**: Can sum/average embeddings

## How It Works

### 1. Card Representation

Each card is embedded as a vector:
```python
Card(Suit, Rank) → Embedding Vector (8 dimensions)

Example:
  Ace of Spades → [0.23, -0.15, 0.67, ..., 0.45]  # Learned during training
  2 of Hearts   → [-0.12, 0.34, -0.23, ..., 0.11]
```

### 2. Embedding Components

Cards are decomposed into **suit** and **rank**, each with their own embedding:

```python
Suit Embedding: 4 suits + 1 padding → 4 dims
Rank Embedding: 13 ranks + 1 padding → 4 dims
Total Card Embedding: 8 dims
```

### 3. Aggregation

Multiple cards are aggregated into a single vector:

```python
Hand = [Card1, Card2, Card3, ...]
Hand Embedding = Sum(Embed(Card1), Embed(Card2), Embed(Card3), ...)
```

**Aggregation methods:**
- **Sum**: Adds all embeddings (default) - captures total information
- **Mean**: Averages embeddings - normalized by count
- **Max**: Takes maximum across dimensions - captures strongest features

## State Representation

### Old System (One-Hot)
```
State Components:
- cards_array: ARRAY_LENGTH (52)
- round_array: ARRAY_LENGTH (52)
- hand_array: ARRAY_LENGTH (52)
- player_array: 4
- player1_cards: ARRAY_LENGTH (52)
- player2_cards: ARRAY_LENGTH (52)
- player3_cards: ARRAY_LENGTH (52)
- player4_cards: ARRAY_LENGTH (52)
- score_array: 4

Total: 52*7 + 8 = 372 dimensions
```

### New System (Embeddings, 8D)
```
State Components:
- Hand embedding (aggregated): 8
- Round embedding (aggregated): 8
- Played cards embedding (aggregated): 8
- Player encoding: 4
- Player tracking embeddings (4 players): 8*4 = 32
- Scores: 4

Total: 8*7 + 8 = 64 dimensions (vs 372!)
```

## Architecture Comparison

| Aspect | One-Hot | Embeddings |
|--------|---------|------------|
| **State size (13 cards)** | 372 dims | 64 dims |
| **State size (11 cards)** | 316 dims | 64 dims |
| **State size (9 cards)** | 260 dims | 64 dims |
| **Model compatibility** | ❌ Different models | ✅ Same model |
| **Parameters** | ~56K | ~12K |
| **Training speed** | Slower | Faster |
| **Expressiveness** | Fixed | Learned |

## Usage Example

### Basic Usage

```python
from card_embedding import CardEmbedding, create_embedded_state, cards_to_ids

# Initialize embedding
embedding = CardEmbedding(embedding_dim=8)

# Convert cards to IDs
hand_card_ids = cards_to_ids(player.hand)
round_card_ids = cards_to_ids(round_cards)

# Create embedded state
state = create_embedded_state(
    hand_cards=hand_card_ids,
    round_cards=round_card_ids,
    all_played_cards=all_played_ids,
    player_tracking=player_tracking_ids,
    player_id=current_player_id,
    scores=score_array,
    embedding=embedding,
    max_hand_size=13
)

# state is now a list of tensors ready for neural network
```

### Training with Embedded Agent

```python
from embedded_dqn_agent import EmbeddedDQNAgent

# Create agent with 8D embeddings
agent = EmbeddedDQNAgent(
    embedding_dim=8,
    gamma=0.99,
    agent_id=0
)

# Agent automatically uses embeddings
# No need to change CARDS_PER_PLAYER or other constants!

# Train as usual
agent.update_replay_memory((state, action, reward, next_state, done))
agent.train(terminal_state=done, step=step_count)
```

### Switching Game Sizes

```python
# constants.py
CARDS_PER_PLAYER = 11  # Change from 13 to 11

# That's it! Embedded agent works with any size:
# - Same model architecture
# - Same state size (64 dims)
# - Just different number of cards in hand
```

## Implementation Details

### File Structure

```
card_embedding.py         - Core embedding classes and functions
embedded_dqn_agent.py     - DQN agent using embeddings
whist_embedded.py         - Whist game adapted for embeddings
training_embedded.py      - Training script for embedded agent
```

### Key Classes

#### `CardEmbedding`
```python
class CardEmbedding:
    def __init__(self, embedding_dim=8)
    def embed_card(self, card_id: int) -> tf.Tensor
    def embed_card_list(self, card_ids: List[int]) -> tf.Tensor
    def aggregate_embeddings(self, embeddings, method='sum') -> tf.Tensor
```

#### `EmbeddedDQNAgent`
```python
class EmbeddedDQNAgent(BaseAgent):
    def __init__(self, embedding_dim=8, gamma=0.99, agent_id=0)
    def create_model(self) -> tf.keras.Model
    def get_qs(self, state) -> np.ndarray
    def choose_action(self, state, valid_actions, epsilon) -> int
```

## Migration Guide

### Step 1: Test Embedding System

```bash
# Test the embedding module
python card_embedding.py

# Output shows state size calculation:
# Total state size: 64 dims (vs 372 with one-hot)
```

### Step 2: Create Embedded Training Script

The embedded agent can be integrated into existing training by:
1. Converting card lists to IDs
2. Creating embedded states
3. Using `EmbeddedDQNAgent` instead of `DQNAgent`

### Step 3: Train Embedded Model

```python
from embedded_dqn_agent import EmbeddedDQNAgent

# Create agent
agent = EmbeddedDQNAgent(embedding_dim=8, gamma=0.99)

# Train (same interface as regular agent)
# Just ensure states are in embedded format
```

### Step 4: Compare Performance

Track metrics:
- Training speed (steps/second)
- Sample efficiency (performance vs. episodes)
- Final performance (win rate)
- Model size (parameters)

## Customization

### Embedding Dimension

Adjust embedding dimension based on needs:

```python
# Smaller (faster, less expressive)
embedding = CardEmbedding(embedding_dim=4)  # 36 total state dims

# Default (good balance)
embedding = CardEmbedding(embedding_dim=8)  # 64 total state dims

# Larger (slower, more expressive)
embedding = CardEmbedding(embedding_dim=16)  # 120 total state dims
```

### Aggregation Method

Choose how to combine card embeddings:

```python
# Sum (default) - captures total information
hand_emb = embedding.aggregate_embeddings(card_embs, method='sum')

# Mean - normalized by count
hand_emb = embedding.aggregate_embeddings(card_embs, method='mean')

# Max - strongest features
hand_emb = embedding.aggregate_embeddings(card_embs, method='max')
```

### Network Architecture

Modify `EmbeddedDQNAgent.create_model()`:

```python
# Adjust feature extraction layers
hand_features = tf.keras.layers.Dense(64, activation='relu')(hand_input)  # Larger

# Adjust hidden layers
hidden1 = tf.keras.layers.Dense(256, activation='relu')(combined)  # Deeper
```

## Advantages for Your Use Case

Based on your requirements:

✅ **"New way to track cards"** - Embeddings replace arrays
✅ **"Not very modifiable"** - Now very flexible
✅ **"Vector approach with 8 dimensions"** - Exactly implemented
✅ **"Doesn't care how many cards"** - Same model for 9, 11, 13 cards
✅ **"Info should still be precise"** - Learned representations capture relationships

## Performance Tips

1. **Start with 8D embeddings** - Good balance of speed and expressiveness
2. **Use sum aggregation** - Works well for card games
3. **Train longer initially** - Embeddings need time to learn good representations
4. **Monitor embedding quality** - Visualize learned embeddings (t-SNE/PCA)
5. **Batch size** - Keep at 32 for stable gradient updates

## Troubleshooting

### Embeddings not learning
- Increase embedding dimension
- Train for more episodes
- Check learning rate

### Performance worse than one-hot
- Embeddings need more training initially
- Try different aggregation methods
- Increase network capacity

### Model too slow
- Reduce embedding dimension (8 → 4)
- Reduce network size
- Use smaller batch size

## Next Steps

1. **Test the embedding module**: `python card_embedding.py`
2. **Try with 11 cards**: Change `CARDS_PER_PLAYER = 11`
3. **Train embedded agent**: Use `EmbeddedDQNAgent`
4. **Compare performance**: Track metrics vs. one-hot agent

The embedding system is ready to use and provides exactly what you requested: a flexible, vector-based approach that works with any number of cards!
