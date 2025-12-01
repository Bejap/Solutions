"""Test script for the trump penalty system.

This tests that agents receive penalties for:
1. Not using trump when opponent is winning and agent has trump (partner not winning)
2. Using unnecessarily high trump when lower trump would win
"""

from Whist.core.whist import Whist
from Whist.core import whist_game as wg
from Whist.agents.ew_strategy import EWStrategy
from Whist.utils.constants import (
    TRUMP_NOT_USED_PENALTY, 
    TRUMP_OVERPLAY_PENALTY
)


def test_trump_not_used_penalty():
    """Test penalty when agent doesn't use trump when opponent is winning."""
    print("Testing trump not used penalty...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names, enable_per_card_reward=False)
    
    # Set up EW strategies
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    game.set_ew_strategies(ew_strategies)
    
    # Set up a specific scenario:
    # Player 1 (East, opponent) leads with Hearts Ace
    # Player 2 (South, agent) has no hearts but has trump (Spades)
    # Player 2 should use trump, but plays a different card
    
    game.reset()
    
    # Manually set up hands for testing
    # Player 0 (North): Some hearts
    # Player 1 (East): Ace of Hearts and other cards
    # Player 2 (South): No hearts, but has Spades (trump) and other cards
    # Player 3 (West): Some hearts
    
    game.players[0].hand = [
        wg.Card('Hearts', '5'),
        wg.Card('Hearts', '6'),
        wg.Card('Clubs', '7')
    ]
    
    game.players[1].hand = [
        wg.Card('Hearts', 'A'),
        wg.Card('Diamonds', '8'),
        wg.Card('Clubs', '9')
    ]
    
    game.players[2].hand = [
        wg.Card('Diamonds', '10'),  # Non-trump, non-led suit
        wg.Card('Spades', '3'),  # Trump
        wg.Card('Spades', '4'),  # Trump
    ]
    
    game.players[3].hand = [
        wg.Card('Hearts', '7'),
        wg.Card('Clubs', '8'),
        wg.Card('Diamonds', '9')
    ]
    
    # Set current player to Player 1 (East) to lead
    game.current_player_idx = 1
    
    # Player 1 leads with Ace of Hearts
    game.step(0)  # Play first card (Ace of Hearts)
    
    # Player 2's turn (South, agent)
    # Hand after sorting: [Diamond 10, Spade 3, Spade 4]
    # Play Diamond 10 instead of trump (action 0)
    state, rewards, done = game.step(0)
    
    # Check if penalty was applied
    if rewards[2] < 0:
        print(f"✓ Agent 2 received penalty: {rewards[2]:.1f} for not using trump")
        assert abs(rewards[2] - TRUMP_NOT_USED_PENALTY) < 0.01, \
            f"Expected penalty {TRUMP_NOT_USED_PENALTY}, got {rewards[2]}"
    else:
        print(f"✗ No penalty applied, reward: {rewards[2]}")
        return False
    
    return True


def test_trump_overplay_penalty():
    """Test penalty when agent uses unnecessarily high trump."""
    print("\nTesting trump overplay penalty...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names, enable_per_card_reward=False)
    
    # Set up EW strategies
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    game.set_ew_strategies(ew_strategies)
    
    game.reset()
    
    # Set up scenario:
    # Player 1 (opponent) leads with Hearts 5
    # Player 2 (agent) can't follow, has multiple trumps
    # Player 2 plays King of Spades when 2 of Spades would win
    
    game.players[0].hand = [
        wg.Card('Hearts', '3'),
        wg.Card('Clubs', '4'),
        wg.Card('Diamonds', '5')
    ]
    
    game.players[1].hand = [
        wg.Card('Hearts', '5'),  # Leads with this
        wg.Card('Diamonds', '6'),
        wg.Card('Clubs', '7')
    ]
    
    game.players[2].hand = [
        wg.Card('Diamonds', '8'),  # Non-trump
        wg.Card('Spades', '2'),  # Low trump that would win
        wg.Card('Spades', 'K'),  # High trump (unnecessarily high)
    ]
    
    game.players[3].hand = [
        wg.Card('Hearts', '6'),
        wg.Card('Clubs', '9'),
        wg.Card('Diamonds', '10')
    ]
    
    # Player 1 leads
    game.current_player_idx = 1
    game.step(0)  # Hearts 5
    
    # Player 2's hand after sorting: [Diamonds 8, Spades 2, Spades K]
    # Play King of Spades (action 2) instead of 2 of Spades (action 1)
    state, rewards, done = game.step(2)
    
    # Check if penalty was applied
    if rewards[2] < 0:
        print(f"✓ Agent 2 received penalty: {rewards[2]:.1f} for trump overplay")
        assert abs(rewards[2] - TRUMP_OVERPLAY_PENALTY) < 0.01, \
            f"Expected penalty {TRUMP_OVERPLAY_PENALTY}, got {rewards[2]}"
    else:
        print(f"✗ No penalty applied, reward: {rewards[2]}")
        return False
    
    return True


def test_no_penalty_when_partner_winning():
    """Test that no penalty is applied when partner is winning."""
    print("\nTesting no penalty when partner is winning...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names, enable_per_card_reward=False)
    
    # Set up EW strategies
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    game.set_ew_strategies(ew_strategies)
    
    game.reset()
    
    # Set up scenario:
    # Player 0 (North, agent's partner) leads and wins with Ace of Hearts
    # Player 2 (South, agent) has no hearts and has trump
    # Player 2 doesn't need to trump since partner is winning
    
    game.players[0].hand = [
        wg.Card('Hearts', 'A'),  # Leads and wins
        wg.Card('Clubs', '5'),
        wg.Card('Diamonds', '6')
    ]
    
    game.players[1].hand = [
        wg.Card('Hearts', '7'),
        wg.Card('Diamonds', '8'),
        wg.Card('Clubs', '9')
    ]
    
    game.players[2].hand = [
        wg.Card('Spades', '3'),  # Trump (doesn't need to use)
        wg.Card('Diamonds', '10'),
        wg.Card('Clubs', '2')
    ]
    
    game.players[3].hand = [
        wg.Card('Hearts', '8'),
        wg.Card('Clubs', '4'),
        wg.Card('Diamonds', '3')
    ]
    
    # Player 0 leads
    game.current_player_idx = 0
    game.step(0)  # Hearts Ace
    
    # Player 1 follows
    game.step(0)  # Hearts 7
    
    # Player 2 plays Diamond 10 (doesn't use trump - partner is winning)
    state, rewards, done = game.step(1)
    
    # Should not receive penalty since partner is winning
    if rewards[2] == 0:
        print(f"✓ No penalty applied when partner winning (reward: {rewards[2]})")
    else:
        print(f"✗ Unexpected reward when partner winning: {rewards[2]}")
        return False
    
    return True


def test_overtrumping_partner_penalty():
    """Test penalty when agent overtrumps partner with unnecessarily high trump."""
    print("\nTesting overtrumping partner penalty...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names, enable_per_card_reward=False)
    
    # Set up EW strategies
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    game.set_ew_strategies(ew_strategies)
    
    game.reset()
    
    # Set up scenario matching the new requirement:
    # Player 2 (South, partner) leads with King of Spades (trump)
    # Player 0 (North, agent) has Ace and Queen of Spades
    # Player 0 plays Ace (unnecessarily high) when Queen would also beat any remaining cards
    
    game.players[0].hand = [
        wg.Card('Diamonds', '5'),
        wg.Card('Spades', 'Q'),  # Lower trump option
        wg.Card('Spades', 'A'),  # High trump (unnecessarily high)
    ]
    
    game.players[1].hand = [
        wg.Card('Clubs', '3'),
        wg.Card('Diamonds', '4'),
        wg.Card('Hearts', '5')
    ]
    
    game.players[2].hand = [
        wg.Card('Spades', 'K'),  # Partner leads with this
        wg.Card('Clubs', '6'),
        wg.Card('Diamonds', '7')
    ]
    
    game.players[3].hand = [
        wg.Card('Clubs', '8'),
        wg.Card('Diamonds', '9'),
        wg.Card('Hearts', '10')
    ]
    
    # Player 2 (South) leads with King of Spades
    game.current_player_idx = 2
    game.step(0)  # King of Spades
    
    # Player 3 (West) plays something
    game.step(0)  # Clubs 8
    
    # Player 0 (North, agent) plays
    # Hand after sorting: [Diamonds 5, Spades Q, Spades A]
    # Should play Queen (action 1) but plays Ace (action 2)
    state, rewards, done = game.step(2)
    
    # Check if penalty was applied
    if rewards[0] < 0:
        print(f"✓ Agent 0 received penalty: {rewards[0]:.1f} for overtrumping partner")
        assert abs(rewards[0] - TRUMP_OVERPLAY_PENALTY) < 0.01, \
            f"Expected penalty {TRUMP_OVERPLAY_PENALTY}, got {rewards[0]}"
    else:
        print(f"✗ No penalty applied, reward: {rewards[0]}")
        return False
    
    return True


def test_taking_trick_from_partner_penalty():
    """Test penalty when agent takes trick from partner who is already winning."""
    print("\nTesting taking trick from winning partner penalty...")
    
    player_names = [1, 2, 3, 4]
    game = Whist(player_names, enable_per_card_reward=False)
    
    # Set up EW strategies
    ew_strategies = {
        1: EWStrategy(2, game),
        3: EWStrategy(4, game)
    }
    game.set_ew_strategies(ew_strategies)
    
    game.reset()
    
    # Set up scenario:
    # Player 2 (South, partner) is winning with highest card
    # Player 0 (North, agent) takes the trick by playing even higher
    # This is wasteful since partner was already winning
    
    game.players[0].hand = [
        wg.Card('Hearts', 'A'),  # Highest hearts - takes from partner
        wg.Card('Hearts', '5'),  # Lower hearts - would let partner win
        wg.Card('Clubs', '3'),
    ]
    
    game.players[1].hand = [
        wg.Card('Hearts', '7'),
        wg.Card('Diamonds', '4'),
        wg.Card('Clubs', '5')
    ]
    
    game.players[2].hand = [
        wg.Card('Hearts', 'K'),  # Partner leads with King (high card)
        wg.Card('Clubs', '6'),
        wg.Card('Diamonds', '7')
    ]
    
    game.players[3].hand = [
        wg.Card('Hearts', '6'),
        wg.Card('Diamonds', '8'),
        wg.Card('Clubs', '9')
    ]
    
    # Player 2 (South) leads with King of Hearts
    game.current_player_idx = 2
    game.step(2)  # King of Hearts (action 2 after sorting: Clubs 6, Diamonds 7, Hearts K)
    
    # Player 3 (West) plays lower hearts
    game.step(2)  # 6 of Hearts (action 2 after sorting: Clubs 9, Diamonds 8, Hearts 6)
    
    # Player 0 (North, agent) plays
    # Hand after sorting: [Clubs 3, Hearts 5, Hearts A]
    # Partner is winning with K. Agent should play 5 but plays Ace (action 2)
    state, rewards, done = game.step(2)
    
    # Check if penalty was applied
    if rewards[0] < 0:
        print(f"✓ Agent 0 received penalty: {rewards[0]:.1f} for taking trick from partner")
        from Whist.utils.constants import PARTNER_OVERPLAY_PENALTY
        assert abs(rewards[0] - PARTNER_OVERPLAY_PENALTY) < 0.01, \
            f"Expected penalty {PARTNER_OVERPLAY_PENALTY}, got {rewards[0]}"
    else:
        print(f"✗ No penalty applied, reward: {rewards[0]}")
        return False
    
    return True


def run_all_tests():
    """Run all trump penalty tests."""
    print("=" * 60)
    print("TRUMP PENALTY SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_trump_not_used_penalty,
        test_trump_overplay_penalty,
        test_no_penalty_when_partner_winning,
        test_overtrumping_partner_penalty,
        test_taking_trick_from_partner_penalty
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("=" * 60)
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
