"""Test script to verify the reward system only gives rewards to agents at positions 0 and 2."""

from whist import Whist
from simple_whist_DQN import DQNAgent
from ew_strategy import EWStrategy
import numpy as np

ARRAY_LENGTH = 13

def test_reward_distribution():
    """Test that rewards are only given to agent positions 0 and 2."""
    print("Testing reward distribution...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names)
    
    # Enable debug logging to see reward details
    game.set_monitoring_level('DEBUG')
    
    # Create agents (only for positions 0 and 2)
    agents = [DQNAgent((ARRAY_LENGTH * 7) + 4 + 4, gamma=0.99) if i in [0, 2] else None for i in range(4)]
    
    # Create EW strategy players
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    
    # Run a few tricks to test reward distribution
    game.reset()
    trick_count = 0
    done = False
    
    print("\nRunning test game for 3 tricks...\n")
    
    while trick_count < 3 and not done:
        for _ in range(4):
            current_player_index = game.current_player_idx
            current_player = game.players[current_player_index]
            agent = agents[current_player_index]
            current_state = game.get_init_state()
            
            valid_actions = [i for i, value in enumerate(game.player_hand(current_player)) if value != 0]
            action_space = len(current_player.hand)
            
            if agent is not None:
                # For testing, just pick random valid action
                action = np.random.choice(valid_actions) if valid_actions else 0
            else:
                # Use strategic play
                ew_strategy = ew_strategies[current_player_index]
                action = ew_strategy.choose_action(current_player, valid_actions)
            
            new_state, rewards, done = game.step(action)
            
            # Check reward distribution
            if rewards != 0:
                print(f"Rewards distributed: {rewards}")
                print(f"  - Position 0 (Agent - North): {rewards[0]}")
                print(f"  - Position 1 (EW Strategy - East): {rewards[1]}")
                print(f"  - Position 2 (Agent - South): {rewards[2]}")
                print(f"  - Position 3 (EW Strategy - West): {rewards[3]}")
                
                # Verify only agents get rewards
                assert rewards[1] == 0, f"ERROR: Position 1 (East) should not receive rewards, got {rewards[1]}"
                assert rewards[3] == 0, f"ERROR: Position 3 (West) should not receive rewards, got {rewards[3]}"
                assert rewards[0] != 0 or rewards[2] != 0, "ERROR: At least one agent should receive a reward"
                
                print("✓ Reward distribution correct: Only agents at positions 0 and 2 received rewards\n")
            
            if len(game.round_list) == 0:  # Trick completed
                trick_count += 1
            
            if done:
                break
    
    # Check final stats
    stats = game.get_reward_stats()
    print("\n=== Final Reward Statistics ===")
    print(f"Agent 0 (North) - Total Rewards: {stats['agent_0_total']}, Tricks Won: {stats['agent_0_wins']}")
    print(f"Agent 2 (South) - Total Rewards: {stats['agent_2_total']}, Tricks Won: {stats['agent_2_wins']}")
    print(f"Total Tricks Completed: {stats['tricks_completed']}")
    
    print("\n✓ Test passed! Reward system correctly gives rewards only to agents at positions 0 and 2")
    print("✓ Monitoring is functional with reward statistics tracking")
    
    return True

if __name__ == "__main__":
    try:
        test_reward_distribution()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
