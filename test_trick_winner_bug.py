"""Test that trick winner is correctly determined, especially with trump."""

from whist import Whist
import whist_game as wg

def test_trick_winner_determination():
    """Test that the correct player wins tricks, especially with trump."""
    print("Testing trick winner determination...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    game.reset()
    
    # Test Case 1: All trump cards - highest should win
    print("\nTest 1: All trump (Spades) - highest rank wins")
    game.round_list = [
        (3, wg.Card('Spades', '6')),  # West: 6 of Spades
        (0, wg.Card('Spades', '5')),  # North: 5 of Spades
        (1, wg.Card('Spades', '8')),  # East: 8 of Spades (WINNER - highest)
        (2, wg.Card('Spades', '3')),  # South: 3 of Spades
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    
    print(f"Cards played:")
    for pid, card in game.round_list:
        print(f"  Player {pid}: {card}")
    print(f"Winner: Player {winner_idx}")
    
    assert winner_idx == 1, f"ERROR: Expected Player 1 (8 of Spades) to win, but Player {winner_idx} won"
    print("✓ Correct winner (Player 1 with 8 of Spades)")
    
    # Test Case 2: Trump beats high non-trump
    print("\nTest 2: Low trump beats high non-trump")
    game.round_list = [
        (0, wg.Card('Hearts', 'A')),  # North: Ace of Hearts (led)
        (1, wg.Card('Hearts', 'K')),  # East: King of Hearts
        (2, wg.Card('Spades', '2')),  # South: 2 of Spades (WINNER - trump)
        (3, wg.Card('Hearts', 'Q')),  # West: Queen of Hearts
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    
    print(f"Cards played:")
    for pid, card in game.round_list:
        print(f"  Player {pid}: {card}")
    print(f"Winner: Player {winner_idx}")
    
    assert winner_idx == 2, f"ERROR: Expected Player 2 (2 of Spades trump) to win, but Player {winner_idx} won"
    print("✓ Correct winner (Player 2 with 2 of Spades trump)")
    
    # Test Case 3: No trump - highest in led suit wins
    print("\nTest 3: No trump - highest card in led suit wins")
    game.round_list = [
        (1, wg.Card('Clubs', '5')),   # East: 5 of Clubs (led)
        (2, wg.Card('Clubs', 'K')),   # South: King of Clubs (WINNER - highest in led suit)
        (3, wg.Card('Diamonds', 'A')), # West: Ace of Diamonds (off-suit, can't win)
        (0, wg.Card('Clubs', '7')),   # North: 7 of Clubs
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    
    print(f"Cards played:")
    for pid, card in game.round_list:
        print(f"  Player {pid}: {card}")
    print(f"Winner: Player {winner_idx}")
    
    assert winner_idx == 2, f"ERROR: Expected Player 2 (King of Clubs) to win, but Player {winner_idx} won"
    print("✓ Correct winner (Player 2 with King of Clubs)")
    
    # Test Case 4: The specific bug from the comment
    print("\nTest 4: Bug case - all Spades with different ranks")
    game.round_list = [
        (3, wg.Card('Spades', '6')),  # West: 6 of Spades
        (0, wg.Card('Spades', '5')),  # North: 5 of Spades  
        (1, wg.Card('Spades', '8')),  # East: 8 of Spades (WINNER - highest)
        (2, wg.Card('Spades', '3')),  # South: 3 of Spades
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    
    print(f"Cards played:")
    for pid, card in game.round_list:
        position_name = ['North', 'East', 'South', 'West'][pid]
        print(f"  {position_name} (Player {pid}): {card}")
    
    winner_name = ['North', 'East', 'South', 'West'][winner_idx]
    print(f"Winner: {winner_name} (Player {winner_idx})")
    
    if winner_idx != 1:
        print(f"✗ ERROR: Expected East (Player 1) with 8 of Spades to win, but {winner_name} (Player {winner_idx}) won")
        print(f"  Rank values: 6={wg.Card('Spades', '6').rank_value}, 5={wg.Card('Spades', '5').rank_value}, 8={wg.Card('Spades', '8').rank_value}, 3={wg.Card('Spades', '3').rank_value}")
        return False
    else:
        print("✓ Correct winner (East with 8 of Spades)")
        return True

if __name__ == "__main__":
    try:
        if test_trick_winner_determination():
            print("\n" + "="*70)
            print("✓ ALL TRICK WINNER TESTS PASSED!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ TRICK WINNER TESTS FAILED!")
            print("="*70)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
