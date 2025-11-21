"""Test that benchmark games with fixed seeds produce identical results."""

from whist import Whist
import numpy as np

def test_fixed_seed_reproducibility():
    """Test that using the same seed produces identical games."""
    print("Testing fixed seed reproducibility...")
    
    player_names = [1, 2, 3, 4]
    
    # Create two game instances
    game1 = Whist(player_names)
    game2 = Whist(player_names)
    
    # Reset both with the same seed
    SEED = 42
    game1.reset(seed=SEED)
    game2.reset(seed=SEED)
    
    # Check that starting player is the same
    assert game1.current_player_idx == game2.current_player_idx, \
        f"Starting players differ: {game1.current_player_idx} vs {game2.current_player_idx}"
    print(f"✓ Both games start with player {game1.current_player_idx}")
    
    # Check that all players have identical hands
    for i in range(4):
        hand1 = sorted([(c.suit, c.rank) for c in game1.players[i].hand])
        hand2 = sorted([(c.suit, c.rank) for c in game2.players[i].hand])
        
        assert hand1 == hand2, f"Player {i} hands differ!\n  Game1: {hand1}\n  Game2: {hand2}"
    
    print("✓ All players have identical hands in both games")
    
    # Test multiple resets with same seed
    print("\nTesting multiple resets with same seed...")
    hands_first_reset = []
    for i in range(4):
        hands_first_reset.append(sorted([(c.suit, c.rank) for c in game1.players[i].hand]))
    
    # Reset again with same seed
    game1.reset(seed=SEED)
    
    for i in range(4):
        hand_after_reset = sorted([(c.suit, c.rank) for c in game1.players[i].hand])
        assert hands_first_reset[i] == hand_after_reset, \
            f"Player {i} hand changed after reset with same seed"
    
    print("✓ Multiple resets with same seed produce identical results")
    
    # Test that different seeds produce different games
    print("\nTesting that different seeds produce different games...")
    game1.reset(seed=42)
    game2.reset(seed=99)
    
    different = False
    for i in range(4):
        hand1 = sorted([(c.suit, c.rank) for c in game1.players[i].hand])
        hand2 = sorted([(c.suit, c.rank) for c in game2.players[i].hand])
        if hand1 != hand2:
            different = True
            break
    
    assert different, "Different seeds should produce different games"
    print("✓ Different seeds produce different games")
    
    # Test that no seed (random) produces different games
    print("\nTesting that no seed produces random games...")
    game1.reset()  # No seed
    game2.reset()  # No seed
    
    different = False
    for i in range(4):
        hand1 = sorted([(c.suit, c.rank) for c in game1.players[i].hand])
        hand2 = sorted([(c.suit, c.rank) for c in game2.players[i].hand])
        if hand1 != hand2:
            different = True
            break
    
    # This should be different (though theoretically could be same by chance)
    if different:
        print("✓ Games without seed are random (produced different results)")
    else:
        print("⚠ Games without seed happened to be identical (rare but possible)")
    
    return True

def test_benchmark_game_scenario():
    """Test a complete benchmark game scenario."""
    print("\n" + "="*70)
    print("Testing complete benchmark game scenario...")
    print("="*70)
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    
    # Simulate games 100, 200, 300 with same seed
    BENCHMARK_SEED = 42
    results = []
    
    for game_num in [100, 200, 300]:
        game.reset(seed=BENCHMARK_SEED)
        
        # Capture game state
        starting_player = game.current_player_idx
        hands = []
        for i in range(4):
            hand = sorted([(c.suit, c.rank) for c in game.players[i].hand])
            hands.append(hand)
        
        results.append({
            'game': game_num,
            'starting_player': starting_player,
            'hands': hands
        })
        
        print(f"\nGame {game_num} (Benchmark):")
        print(f"  Starting player: {starting_player}")
        print(f"  North hand size: {len(hands[0])} cards")
        print(f"  First card: {hands[0][0] if hands[0] else 'None'}")
    
    # Verify all benchmark games are identical
    print("\nVerifying all benchmark games are identical...")
    assert results[0]['starting_player'] == results[1]['starting_player'] == results[2]['starting_player'], \
        "Starting players should be identical"
    
    for player_idx in range(4):
        assert results[0]['hands'][player_idx] == results[1]['hands'][player_idx] == results[2]['hands'][player_idx], \
            f"Player {player_idx} hands should be identical across benchmark games"
    
    print("✓ All benchmark games (100, 200, 300) are identical!")
    print("✓ This allows tracking model improvement on the exact same scenario")
    
    return True

if __name__ == "__main__":
    try:
        test_fixed_seed_reproducibility()
        test_benchmark_game_scenario()
        
        print("\n" + "="*70)
        print("✓ ALL BENCHMARK TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("- Fixed seeds produce identical games")
        print("- Every 100th game (100, 200, 300...) will use seed 42")
        print("- This allows tracking if the model improves on the same scenario")
        print("- Other games remain random for diverse training")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
