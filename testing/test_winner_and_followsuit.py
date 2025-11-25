"""Test that follow suit is enforced and winner starts next trick."""

import os
import shutil
from Whist.core.whist import Whist
from Whist.agents.simple_whist_DQN import DQNAgent
from Whist.agents.ew_strategy import EWStrategy
from Whist.logger.game_logger import GameLogger
import numpy as np

ARRAY_LENGTH = 52

def test_follow_suit_enforcement():
    """Test that EW strategy respects valid actions and can't play invalid cards."""
    print("Testing follow suit enforcement in EW strategy...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    game.reset()
    
    # Create EW strategy players
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    
    # Test over multiple tricks to ensure follow suit is respected
    tricks_tested = 0
    violations = []
    
    for trick in range(5):
        if not game.players[0].hand:
            break
            
        for card_in_trick in range(4):
            current_player_idx = game.current_player_idx
            current_player = game.players[current_player_idx]
            
            if not current_player.hand:
                break
            
            # Get valid actions
            valid_actions = game.get_valid_actions(current_player)
            
            # Choose action based on player type
            if current_player_idx in [1, 3]:
                # EW strategy player
                action = ew_strategies[current_player_idx].choose_action(current_player, valid_actions)
            else:
                # Random for agents in this test
                action = np.random.choice(valid_actions) if valid_actions else 0
            
            # Verify action is valid
            if action not in valid_actions:
                current_player._sort_hand()
                played_card = current_player.hand[action]
                led_suit = game.round_list[0][1].suit if game.round_list else None
                violations.append({
                    'trick': trick + 1,
                    'player': current_player_idx,
                    'card': str(played_card),
                    'led_suit': led_suit,
                    'valid_actions': valid_actions,
                    'chosen_action': action
                })
            
            # Execute the move
            game.step(action)
            
            if len(game.round_list) == 0:  # Trick completed
                tricks_tested += 1
                break
    
    print(f"✓ Tested {tricks_tested} tricks")
    
    if violations:
        print(f"\n✗ Found {len(violations)} follow suit violations:")
        for v in violations:
            print(f"  Trick {v['trick']}, Player {v['player']}: Played {v['card']}")
            print(f"    Led suit: {v['led_suit']}, Valid actions: {v['valid_actions']}, Chose: {v['chosen_action']}")
        return False
    else:
        print("✓ No follow suit violations detected!")
        return True

def test_winner_starts_next_trick():
    """Test that the winner of a trick starts the next trick."""
    print("\nTesting that winner starts next trick...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    
    # Create agents and strategies
    agents = [DQNAgent((ARRAY_LENGTH * 7) + 4 + 4, gamma=0.99) if i in [0, 2] else None for i in range(4)]
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    
    # Initialize game logger for detailed output
    test_log_dir = 'test_winner_starts'
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)
    
    game_logger = GameLogger(log_dir=test_log_dir)
    
    game.reset()
    starting_player = game.current_player_idx
    print(f"Game starts with player {starting_player}")
    
    game_logger.start_game(1, starting_player, game.players, trump_suit='Spades')
    
    trick_leaders = [starting_player]
    trick_winners = []
    
    # Play 3 complete tricks
    for trick_num in range(3):
        print(f"\nTrick {trick_num + 1} - Leader: Player {game.current_player_idx}")
        
        for card_in_trick in range(4):
            current_player_idx = game.current_player_idx
            current_player = game.players[current_player_idx]
            
            if not current_player.hand:
                break
            
            # Get valid actions
            valid_actions = game.get_valid_actions(current_player)
            
            # Choose action
            agent = agents[current_player_idx]
            if agent is not None:
                action = np.random.choice(valid_actions) if valid_actions else 0
                decision_type = 'random'
                certainty = None
            else:
                ew_strategy = ew_strategies[current_player_idx]
                action = ew_strategy.choose_action(current_player, valid_actions)
                decision_type = 'strategy'
                certainty = None
            
            # Log the card
            current_player._sort_hand()
            played_card = current_player.hand[action]
            game_logger.log_card_played(current_player_idx, played_card, decision_type, certainty)
            
            # Execute move
            new_state, rewards, done = game.step(action)
            
            if len(game.round_list) == 0:  # Trick completed
                winner_idx = game.players.index(game.trick_winner)
                trick_winners.append(winner_idx)
                print(f"  Winner: Player {winner_idx}")
                
                # Complete trick logging
                game_logger.complete_trick(winner_idx)
                
                # Record next trick's leader (should be the winner)
                if trick_num < 2:  # Not the last trick
                    next_leader = game.current_player_idx
                    trick_leaders.append(next_leader)
                    print(f"  Next leader will be: Player {next_leader}")
                break
    
    # Save the log
    game_logger.end_game(1, game.score_array)
    
    # Verify winners match leaders
    print(f"\nVerification:")
    print(f"Trick winners: {trick_winners}")
    print(f"Trick leaders: {trick_leaders}")
    
    success = True
    for i in range(len(trick_winners)):
        if i + 1 < len(trick_leaders):  # Check if winner leads next trick
            winner = trick_winners[i]
            next_leader = trick_leaders[i + 1]
            if winner == next_leader:
                print(f"✓ Trick {i + 1}: Winner {winner} correctly leads trick {i + 2}")
            else:
                print(f"✗ Trick {i + 1}: Winner {winner} should lead trick {i + 2}, but {next_leader} leads instead")
                success = False
    
    # Show the log
    log_file = os.path.join(test_log_dir, "game_0001.txt")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
        print(f"\n{'='*70}")
        print("GAME LOG:")
        print('='*70)
        print(log_content)
        print('='*70)
    
    # Cleanup
    shutil.rmtree(test_log_dir)
    
    return success

if __name__ == "__main__":
    try:
        result1 = test_follow_suit_enforcement()
        result2 = test_winner_starts_next_trick()
        
        print("\n" + "="*70)
        if result1 and result2:
            print("✓ ALL TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED")
        print("="*70)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
