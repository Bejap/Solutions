"""Test script to verify the trump system works correctly with Spades as trump."""

import whist_game as wg

def test_trump_card_identification():
    """Test that trump cards are correctly identified."""
    print("Testing trump card identification...")
    
    # Test all suits
    cards = [
        wg.Card('Clubs', '5'),
        wg.Card('Diamonds', 'K'),
        wg.Card('Hearts', 'A'),
        wg.Card('Spades', '2'),  # Trump
        wg.Card('Spades', 'A'),  # Trump
    ]
    
    for card in cards:
        is_trump = card.is_trump()
        expected = (card.suit == 'Spades')
        assert is_trump == expected, f"ERROR: {card} trump check failed"
        print(f"  {card}: Trump={is_trump} ✓")
    
    print("✓ Trump card identification works correctly\n")

def test_trump_system():
    """Test that trump cards win tricks correctly."""
    print("Testing trump system in trick evaluation...")
    
    from whist import Whist
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    game.set_monitoring_level('DEBUG')
    
    # Manually set up a trick scenario
    # Scenario 1: Trump (Spades) beats high non-trump card
    print("\n--- Scenario 1: Low trump beats high non-trump ---")
    game.round_list = [
        (0, wg.Card('Hearts', 'A')),    # Player 0 leads with Ace of Hearts
        (1, wg.Card('Hearts', 'K')),    # Player 1 follows with King of Hearts
        (2, wg.Card('Spades', '2')),    # Player 2 plays 2 of Spades (TRUMP - should win!)
        (3, wg.Card('Hearts', '5')),    # Player 3 follows with 5 of Hearts
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    print(f"Winner: Player {winner_idx}")
    assert winner_idx == 2, f"ERROR: Expected Player 2 (trump) to win, but Player {winner_idx} won"
    print("✓ Low trump correctly beats high non-trump cards\n")
    
    # Scenario 2: Highest trump wins when multiple trumps played
    print("--- Scenario 2: Highest trump wins among multiple trumps ---")
    game.round_list = [
        (0, wg.Card('Spades', '5')),    # Player 0 leads with 5 of Spades (trump)
        (1, wg.Card('Spades', 'K')),    # Player 1 plays King of Spades (trump - should win!)
        (2, wg.Card('Spades', '7')),    # Player 2 plays 7 of Spades (trump)
        (3, wg.Card('Hearts', 'A')),    # Player 3 can't follow, plays Ace of Hearts
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    print(f"Winner: Player {winner_idx}")
    assert winner_idx == 1, f"ERROR: Expected Player 1 (King of Spades) to win, but Player {winner_idx} won"
    print("✓ Highest trump correctly wins among multiple trumps\n")
    
    # Scenario 3: No trump played - highest card in led suit wins
    print("--- Scenario 3: No trump - highest card in led suit wins ---")
    game.round_list = [
        (0, wg.Card('Hearts', '5')),    # Player 0 leads with 5 of Hearts
        (1, wg.Card('Hearts', 'A')),    # Player 1 plays Ace of Hearts (should win!)
        (2, wg.Card('Hearts', '7')),    # Player 2 follows with 7 of Hearts
        (3, wg.Card('Hearts', 'K')),    # Player 3 follows with King of Hearts
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    print(f"Winner: Player {winner_idx}")
    assert winner_idx == 1, f"ERROR: Expected Player 1 (Ace of Hearts) to win, but Player {winner_idx} won"
    print("✓ Highest card in led suit wins when no trump played\n")
    
    # Scenario 4: Off-suit cards don't win (only led suit or trump)
    print("--- Scenario 4: Off-suit cards don't win (must follow suit) ---")
    game.round_list = [
        (0, wg.Card('Hearts', '2')),     # Player 0 leads with 2 of Hearts
        (1, wg.Card('Diamonds', 'A')),   # Player 1 plays Ace of Diamonds (off-suit)
        (2, wg.Card('Hearts', '3')),     # Player 2 follows with 3 of Hearts (should win!)
        (3, wg.Card('Clubs', 'K')),      # Player 3 plays King of Clubs (off-suit)
    ]
    
    winner = game._evaluate_trick_winner()
    winner_idx = game.players.index(winner)
    print(f"Winner: Player {winner_idx}")
    assert winner_idx == 2, f"ERROR: Expected Player 2 (3 of Hearts) to win, but Player {winner_idx} won"
    print("✓ Only led suit or trump can win tricks\n")
    
    print("✓ All trump system tests passed!\n")

def test_full_deck():
    """Test that the full 52-card deck is created correctly."""
    print("Testing full deck composition...")
    
    deck = wg.Deck()
    cards = deck.get_deck()
    
    print(f"Total cards: {len(cards)}")
    assert len(cards) == 52, f"ERROR: Expected 52 cards, got {len(cards)}"
    
    # Count cards by suit
    suit_counts = {}
    trump_count = 0
    for card in cards:
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        if card.is_trump():
            trump_count += 1
    
    print(f"Cards by suit:")
    for suit, count in sorted(suit_counts.items()):
        trump_marker = " (TRUMP)" if suit == wg.Card.TRUMP_SUIT else ""
        print(f"  {suit}: {count} cards{trump_marker}")
    
    assert len(suit_counts) == 4, f"ERROR: Expected 4 suits, got {len(suit_counts)}"
    for suit, count in suit_counts.items():
        assert count == 13, f"ERROR: Expected 13 cards in {suit}, got {count}"
    
    assert trump_count == 13, f"ERROR: Expected 13 trump cards, got {trump_count}"
    print(f"\n✓ Full 52-card deck correctly created with Spades as trump\n")

if __name__ == "__main__":
    try:
        test_trump_card_identification()
        test_full_deck()
        test_trump_system()
        print("\n" + "="*60)
        print("✓ ALL TRUMP SYSTEM TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
