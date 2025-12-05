"""Test script to verify game logging functionality."""

import os
import shutil
from Whist.core.whist import Whist
from Whist.logger.game_logger import GameLogger
from Whist.core import whist_game as wg

def test_game_logger():
    """Test that game logger creates proper .txt files with correct format."""
    print("Testing game logger...")
    
    # Create a test log directory
    test_log_dir = 'test_game_logs'
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)
    
    # Initialize game and logger
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    logger = GameLogger(log_dir=test_log_dir)
    
    # Reset game to get starting hands
    game.reset()
    starting_player_idx = game.current_player_idx
    
    # Start logging
    episode = 50
    logger.start_game(episode, starting_player_idx, game.players)
    
    print(f"\n✓ Started logging game {episode}")
    print(f"✓ Starting player: {logger.POSITION_NAMES[starting_player_idx]}")
    
    # Play a few tricks
    tricks_to_play = 3
    for trick in range(tricks_to_play):
        for _ in range(4):
            current_player = game.players[game.current_player_idx]
            if not current_player.hand:
                break
            
            # Play first valid card
            current_player._sort_hand()
            card = current_player.hand[0]
            
            # Simulate decision type based on player position
            player_idx = game.current_player_idx
            if player_idx in [0, 2]:  # Agents
                # Alternate between agent decision and random
                if trick % 2 == 0:
                    decision_type = 'agent'
                    certainty = 0.8 + (trick * 0.05)  # Increasing confidence
                else:
                    decision_type = 'random'
                    certainty = None
            else:  # Strategy players
                decision_type = 'strategy'
                certainty = None
            
            # Log the card with decision info
            logger.log_card_played(game.current_player_idx, card, decision_type, certainty)
            
            # Execute the move
            game.step(0)
            
            if len(game.round_list) == 0:  # Trick complete
                # Get the trick winner and log it
                if game.trick_winner:
                    winner_idx = game.players.index(game.trick_winner)
                    logger.complete_trick(winner_idx)
                break
    
    print(f"✓ Logged {tricks_to_play} tricks")
    
    # End the game
    logger.end_game(episode, game.score_array)
    
    # Verify the file was created
    log_file = os.path.join(test_log_dir, f"game_{episode:04d}.txt")
    assert os.path.exists(log_file), f"ERROR: Log file {log_file} was not created"
    
    print(f"✓ Log file created: {log_file}")
    
    # Read and display the log
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    print(f"\n{'='*70}")
    print("GAME LOG CONTENT:")
    print('='*70)
    print(log_content)
    print('='*70)
    
    # Verify log format
    assert "=== Game 50 ===" in log_content, "ERROR: Missing game header"
    assert "Starting player:" in log_content, "ERROR: Missing starting player"
    assert "North hand:" in log_content, "ERROR: Missing North hand"
    assert "East hand:" in log_content, "ERROR: Missing East hand"
    assert "South hand:" in log_content, "ERROR: Missing South hand"
    assert "West hand:" in log_content, "ERROR: Missing West hand"
    assert "Trick 1:" in log_content, "ERROR: Missing trick information"
    assert "N:" in log_content or "E:" in log_content or "S:" in log_content or "W:" in log_content, "ERROR: Missing card plays"
    assert "Final Scores:" in log_content, "ERROR: Missing final scores"
    
    # Verify decision information is present
    assert "[Agent:" in log_content or "[Random" in log_content or "[Strategy]" in log_content, "ERROR: Missing decision information"
    
    print("\n✓ All format checks passed")
    print("✓ Decision information included in logs")
    print("✓ Game logger test completed successfully!")
    
    # Cleanup
    shutil.rmtree(test_log_dir)
    print(f"✓ Cleaned up test directory")
    
    return True

if __name__ == "__main__":
    try:
        test_game_logger()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
