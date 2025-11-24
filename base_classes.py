"""
Base Classes for Whist DQN Project

This module provides abstract base classes that define the inheritance hierarchy
for the entire project, enabling better code organization and extensibility.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional, Any


class BaseAgent(ABC):
    """Abstract base class for all agents (DQN and strategy-based)."""
    
    def __init__(self, agent_id: int):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
    
    @abstractmethod
    def choose_action(self, state: Any, valid_actions: List[int], epsilon: Optional[float] = None) -> int:
        """
        Choose an action based on the current state.
        
        Args:
            state: The current game state
            valid_actions: List of valid action indices
            epsilon: Optional exploration rate (for epsilon-greedy agents)
            
        Returns:
            The chosen action index
        """
        pass
    
    @abstractmethod
    def update(self, transition: Tuple) -> None:
        """
        Update the agent based on a state transition.
        
        Args:
            transition: Tuple of (state, action, reward, next_state, done)
        """
        pass


class BasePlayer(ABC):
    """Abstract base class for all player types."""
    
    def __init__(self, player_id: int):
        """
        Initialize the base player.
        
        Args:
            player_id: Unique identifier for the player (1-4)
        """
        self.id = player_id
        self.hand = []
        self.tricks_won = 0
        self.last_played_card = None
        self.known_actions = []
    
    @abstractmethod
    def action(self, choice: int):
        """
        Execute an action (play a card).
        
        Args:
            choice: The index of the card to play
            
        Returns:
            The card that was played
        """
        pass
    
    def observe(self, player_id: int, action: Any) -> None:
        """
        Observe another player's action.
        
        Args:
            player_id: The ID of the player who took the action
            action: The action taken (e.g., card rank value)
        """
        if player_id != self.id:
            self.known_actions.append((player_id, action))
    
    def resetting_observation(self) -> None:
        """Reset the player's observations at the start of a new game."""
        self.known_actions = []
    
    @abstractmethod
    def get_hand(self):
        """
        Get the player's hand.
        
        Returns:
            The player's current hand
        """
        pass


class BaseStrategy(ABC):
    """Abstract base class for all playing strategies.
    
    Note: BaseStrategy and BaseAgent have different choose_action signatures by design:
    - BaseAgent.choose_action(state, valid_actions, epsilon) - works with encoded states
    - BaseStrategy.choose_action(player, valid_actions) - works with player objects
    
    This is intentional: agents learn from abstract states, while strategies need
    to inspect the actual player object (hand, observations, etc.) to make decisions.
    They are not meant to be polymorphic with each other, as they serve different purposes.
    """
    
    def __init__(self, player_id: int):
        """
        Initialize the base strategy.
        
        Args:
            player_id: The ID of the player using this strategy
        """
        self.player_id = player_id
    
    @abstractmethod
    def choose_action(self, player: BasePlayer, valid_actions: List[int]) -> int:
        """
        Choose an action based on the strategy.
        
        Args:
            player: The player object
            valid_actions: List of valid action indices
            
        Returns:
            The chosen action index
        """
        pass


class BaseGame(ABC):
    """Abstract base class for card game implementations."""
    
    def __init__(self, num_players: int):
        """
        Initialize the base game.
        
        Args:
            num_players: Number of players in the game
        """
        self.num_players = num_players
        self.players = []
        self.current_player_idx = 0
        self.trick_winner = None
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """
        Reset the game to initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            The initial game state
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[Any, Any, bool]:
        """
        Execute one step of the game.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (new_state, reward, done)
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, player: BasePlayer) -> List[int]:
        """
        Get valid actions for a player.
        
        Args:
            player: The player to get valid actions for
            
        Returns:
            List of valid action indices
        """
        pass
    
    @abstractmethod
    def deal_cards(self) -> None:
        """Deal cards to all players."""
        pass


class BaseCard(ABC):
    """Abstract base class for playing cards."""
    
    def __init__(self, suit: str, rank: str):
        """
        Initialize a card.
        
        Args:
            suit: The suit of the card
            rank: The rank of the card
        """
        self.suit = suit
        self.rank = rank
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the card."""
        pass
    
    @abstractmethod
    def compare_to(self, other: 'BaseCard', led_suit: Optional[str] = None) -> int:
        """
        Compare this card to another card.
        
        Args:
            other: The other card to compare to
            led_suit: The suit that was led (for trick-taking games)
            
        Returns:
            Positive if this card is higher, negative if lower, 0 if equal
        """
        pass


class BaseDeck(ABC):
    """Abstract base class for card decks."""
    
    def __init__(self):
        """Initialize an empty deck."""
        self.card_deck = []
    
    @abstractmethod
    def shuffle(self) -> None:
        """Shuffle the deck."""
        pass
    
    @abstractmethod
    def deal(self, num_cards: int) -> List[BaseCard]:
        """
        Deal cards from the deck.
        
        Args:
            num_cards: Number of cards to deal
            
        Returns:
            List of dealt cards
        """
        pass
    
    def get_deck(self) -> List[BaseCard]:
        """
        Get the current deck.
        
        Returns:
            The current deck of cards
        """
        return self.card_deck
