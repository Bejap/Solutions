"""
Demonstration of the Enhanced Inheritance Structure

This script demonstrates the new inheritance hierarchy and polymorphic behavior
of the refactored Whist DQN codebase.
"""

from Whist.utils.base_classes import (
    BaseAgent, BasePlayer, BaseGame, BaseStrategy, 
    BaseCard, BaseDeck
)
from Whist.core.whist_game import Card, Deck, Player
from Whist.core.whist import Whist
from Whist.agents.simple_whist_DQN import DQNAgent
from Whist.agents.ew_strategy import EWStrategy
from Whist.utils.constants import ARRAY_LENGTH, NUM_PLAYERS, DQN_AGENT_POSITIONS


def demonstrate_inheritance():
    """Demonstrate the inheritance relationships."""
    print("=" * 80)
    print("WHIST DQN - INHERITANCE STRUCTURE DEMONSTRATION")
    print("=" * 80)
    print()
    
    # 1. Card Hierarchy
    print("1. CARD HIERARCHY (BaseCard → Card)")
    print("-" * 80)
    card1 = Card('Spades', 'A')
    card2 = Card('Hearts', 'K')
    print(f"   Created cards: {card1}, {card2}")
    print(f"   Card is BaseCard: {isinstance(card1, BaseCard)}")
    print(f"   Card has compare_to: {hasattr(card1, 'compare_to')}")
    
    # Test comparison
    comparison = card1.compare_to(card2, led_suit='Hearts')
    print(f"   Comparing {card1} vs {card2} (Hearts led): {comparison}")
    print()
    
    # 2. Deck Hierarchy
    print("2. DECK HIERARCHY (BaseDeck → Deck)")
    print("-" * 80)
    deck = Deck()
    print(f"   Created deck with {len(deck.get_deck())} cards")
    print(f"   Deck is BaseDeck: {isinstance(deck, BaseDeck)}")
    deck.shuffle()
    print(f"   Deck shuffled successfully")
    dealt_cards = deck.deal(5)
    print(f"   Dealt {len(dealt_cards)} cards: {dealt_cards}")
    print()
    
    # 3. Player Hierarchy
    print("3. PLAYER HIERARCHY (BasePlayer → Player)")
    print("-" * 80)
    player = Player(1)
    print(f"   Created Player {player.id}")
    print(f"   Player is BasePlayer: {isinstance(player, BasePlayer)}")
    print(f"   Player attributes: hand={len(player.hand)}, tricks_won={player.tricks_won}")
    
    # Test observation
    player.observe(2, 14)
    print(f"   Player observed action: {player.known_actions}")
    player.resetting_observation()
    print(f"   Observations reset: {player.known_actions}")
    print()
    
    # 4. Game Hierarchy
    print("4. GAME HIERARCHY (BaseGame → Whist)")
    print("-" * 80)
    game = Whist([1, 2, 3, 4])
    print(f"   Created Whist game with {len(game.players)} players")
    print(f"   Whist is BaseGame: {isinstance(game, BaseGame)}")
    print(f"   Game has required methods:")
    print(f"     - reset: {hasattr(game, 'reset')}")
    print(f"     - step: {hasattr(game, 'step')}")
    print(f"     - get_valid_actions: {hasattr(game, 'get_valid_actions')}")
    print(f"     - deal_cards: {hasattr(game, 'deal_cards')}")
    print()
    
    # 5. Agent Hierarchy
    print("5. AGENT HIERARCHY (BaseAgent → DQNAgent)")
    print("-" * 80)
    input_size = (ARRAY_LENGTH * 7) + 4 + 4
    agent = DQNAgent(input_size, gamma=0.99, agent_id=0)
    print(f"   Created DQNAgent with ID {agent.agent_id}")
    print(f"   DQNAgent is BaseAgent: {isinstance(agent, BaseAgent)}")
    print(f"   Agent model input shape: {input_size}")
    print(f"   Agent has required methods:")
    print(f"     - choose_action: {hasattr(agent, 'choose_action')}")
    print(f"     - update: {hasattr(agent, 'update')}")
    print(f"     - get_qs: {hasattr(agent, 'get_qs')}")
    print()
    
    # 6. Strategy Hierarchy
    print("6. STRATEGY HIERARCHY (BaseStrategy → EWStrategy)")
    print("-" * 80)
    ew_strategy = EWStrategy(2, game)
    print(f"   Created EWStrategy for player ID {ew_strategy.player_id}")
    print(f"   EWStrategy is BaseStrategy: {isinstance(ew_strategy, BaseStrategy)}")
    print(f"   EWStrategy is NOT BaseAgent (by design): {isinstance(ew_strategy, BaseAgent)}")
    print(f"   Note: Strategies and Agents have different interfaces intentionally")
    print(f"   Random play probability: {ew_strategy.random_play_probability}")
    print()
    
    # 7. Polymorphism Demonstration
    print("7. POLYMORPHISM DEMONSTRATION")
    print("-" * 80)
    
    # All agents can use choose_action interface
    print("   All agents implement choose_action:")
    agents_list = [
        (DQNAgent(input_size, gamma=0.99, agent_id=i), f"DQNAgent({i})") 
        for i in DQN_AGENT_POSITIONS
    ]
    strategies_list = [
        (EWStrategy(i+1, game), f"EWStrategy({i+1})") 
        for i in [1, 3]
    ]
    
    for agent_obj, name in agents_list:
        has_method = hasattr(agent_obj, 'choose_action')
        is_base = isinstance(agent_obj, BaseAgent)
        print(f"     {name}: has_choose_action={has_method}, is_BaseAgent={is_base}")
    
    for strategy_obj, name in strategies_list:
        has_method = hasattr(strategy_obj, 'choose_action')
        is_base = isinstance(strategy_obj, BaseStrategy)
        print(f"     {name}: has_choose_action={has_method}, is_BaseStrategy={is_base}")
    print()
    
    # 8. Summary
    print("8. INHERITANCE STRUCTURE SUMMARY")
    print("-" * 80)
    print("   ✓ BaseCard → Card")
    print("   ✓ BaseDeck → Deck")
    print("   ✓ BasePlayer → Player")
    print("   ✓ BaseGame → Whist")
    print("   ✓ BaseAgent → DQNAgent")
    print("   ✓ BaseStrategy → EWStrategy")
    print()
    print("   Benefits:")
    print("   • Polymorphic behavior across all agent types")
    print("   • Easy extensibility for new strategies/agents")
    print("   • Type safety with abstract base classes")
    print("   • Clean separation of concerns")
    print("   • Centralized configuration in constants.py")
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_inheritance()
