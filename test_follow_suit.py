"""Test follow suit rule and updated game logging."""

import os
import shutil
from whist import Whist
from game_logger import GameLogger
import whist_game as wg

def test_follow_suit_rule():
    """Test that players must follow suit when they have cards of the led suit."""
    print("Testing follow suit rule...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    game.reset()
    
    # Simulate a trick where a suit is led
    player = game.players[0]
    
    # Give player some specific cards for testing
    player.hand = [
        wg.Card('Hearts', '5'),
        wg.Card('Hearts', '8'),
        wg.Card('Spades', 'A'),
        wg.Card('Clubs', '3')
    ]
    
    # No cards played yet - all cards should be valid
    valid_actions = game.get_valid_actions(player)
    assert len(valid_actions) == 4, f"ERROR: Expected 4 valid actions at start, got {len(valid_actions)}"
    print(f"✓ Before trick starts: All {len(valid_actions)} cards are valid")
    
    # Simulate Hearts being led
    game.round_list = [(0, wg.Card('Hearts', 'K'))]
    
    # Now only Hearts should be valid
    valid_actions = game.get_valid_actions(player)
    player._sort_hand()
    valid_cards = [player.hand[i] for i in valid_actions]
    
    print(f"✓ After Hearts led: {len(valid_actions)} valid actions")
    for card in valid_cards:
        print(f"  - {card}")
        assert card.suit == 'Hearts', f"ERROR: Only Hearts should be valid, but got {card}"
    
    assert len(valid_actions) == 2, f"ERROR: Expected 2 Hearts to be valid, got {len(valid_actions)}"
    
    # Test when player has no cards of led suit
    player.hand = [
        wg.Card('Spades', 'A'),
        wg.Card('Clubs', '3'),
        wg.Card('Diamonds', '7')
    ]
    
    game.round_list = [(0, wg.Card('Hearts', 'K'))]
    valid_actions = game.get_valid_actions(player)
    
    assert len(valid_actions) == 3, f"ERROR: Expected all 3 cards valid when can't follow, got {len(valid_actions)}"
    print(f"✓ When can't follow suit: All {len(valid_actions)} cards are valid")
    
    print("\n✓ Follow suit rule working correctly!")
    return True

def test_game_logger_with_winner():
    """Test that game logger includes trump and trick winner."""
    print("\nTesting game logger with trump and winner...")
    
    # Create a test log directory
    test_log_dir = 'test_game_logs_winner'
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)
    
    # Initialize game and logger
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    logger = GameLogger(log_dir=test_log_dir)
    
    # Reset game to get starting hands
    game.reset()
    starting_player_idx = game.current_player_idx
    
    # Start logging with trump
    episode = 300
    logger.start_game(episode, starting_player_idx, game.players, trump_suit='Spades')
    
    print(f"✓ Started logging game {episode} with trump: Spades")
    
    # Play one complete trick
    for _ in range(4):
        current_player = game.players[game.current_player_idx]
        if not current_player.hand:
            break
        
        # Play first valid card
        valid_actions = game.get_valid_actions(current_player)
        action = valid_actions[0]
        
        current_player._sort_hand()
        card = current_player.hand[action]
        
        # Log the card with decision info
        logger.log_card_played(game.current_player_idx, card, 'strategy', None)
        
        # Execute the move
        game.step(action)
    
    # Complete the trick with winner
    if game.trick_winner:
        winner_idx = game.players.index(game.trick_winner)
        logger.complete_trick(winner_idx)
        print(f"✓ Trick completed with winner: {logger.POSITION_NAMES[winner_idx]}")
    
    # End the game
    logger.end_game(episode, game.score_array)
    
    # Verify the file was created
    log_file = os.path.join(test_log_dir, f"game_{episode:04d}.txt")
    assert os.path.exists(log_file), f"ERROR: Log file {log_file} was not created"
    
    # Read and verify content
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    print(f"\n{'='*70}")
    print("GAME LOG CONTENT:")
    print('='*70)
    print(log_content)
    print('='*70)
    
    # Verify trump is displayed
    assert "Trump: Spades" in log_content, "ERROR: Trump not displayed in log"
    print("\n✓ Trump suit displayed in log")
    
    # Verify winner is displayed
    assert "winner" in log_content, "ERROR: Trick winner not displayed in log"
    print("✓ Trick winner displayed in log")
    
    # Cleanup
    shutil.rmtree(test_log_dir)
    print("✓ Test passed!\n")
    
    return True

if __name__ == "__main__":
    try:
        test_follow_suit_rule()
        test_game_logger_with_winner()
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
