"""
Prioritized Experience Replay

This module implements prioritized experience replay for DQN agents.
Experiences are sampled based on their TD-error (importance), allowing
the agent to learn more efficiently from surprising transitions.

Based on: "Prioritized Experience Replay" (Schaul et al., 2015)
https://arxiv.org/abs/1511.05952
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import namedtuple


# Experience tuple for cleaner code
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    
    A binary tree where each node contains the sum of its children.
    Leaf nodes contain priorities, internal nodes contain sums.
    This allows O(log n) sampling and O(log n) priority updates.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize Sum Tree.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaf nodes
        self.data = np.zeros(capacity, dtype=object)  # Store experiences
        self.write_index = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve leaf index for a given cumulative sum.
        
        Args:
            idx: Current node index
            s: Target cumulative sum
            
        Returns:
            Leaf index containing the target sum
        """
        left = 2 * idx + 1
        right = left + 1
        
        # If we're at a leaf, return it
        if left >= len(self.tree):
            return idx
        
        # Traverse left or right based on cumulative sum
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return sum of all priorities."""
        return self.tree[0]
    
    def add(self, priority: float, data: Experience):
        """
        Add experience with given priority.
        
        Args:
            priority: Priority value (typically TD-error)
            data: Experience tuple
        """
        idx = self.write_index + self.capacity - 1  # Leaf index
        
        self.data[self.write_index] = data
        self.update(idx, priority)
        
        self.write_index = (self.write_index + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """
        Update priority of experience.
        
        Args:
            idx: Tree index (not data index)
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Experience]:
        """
        Get experience for a given cumulative sum.
        
        Args:
            s: Cumulative sum value
            
        Returns:
            Tuple of (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    
    def __len__(self):
        return self.n_entries


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Memory.
    
    Samples experiences based on their priority (TD-error magnitude).
    Uses importance sampling weights to correct for bias introduced
    by non-uniform sampling.
    
    Key features:
    - Proportional prioritization: P(i) = p_i^α / Σ_k p_k^α
    - Importance sampling weights: w_i = (N * P(i))^(-β)
    - Annealing β from initial value to 1.0 over training
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_frames: int = 100000,
                 epsilon: float = 0.01):
        """
        Initialize prioritized replay memory.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant to ensure non-zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0  # Track maximum priority for new experiences
    
    def _get_beta(self) -> float:
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def append(self, transition: Tuple):
        """
        Add new experience with maximum priority.
        
        Args:
            transition: Tuple of (state, action, reward, next_state, done)
        """
        experience = Experience(*transition)
        # New experiences get maximum priority to ensure they're seen at least once
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, importance_weights)
            - experiences: List of Experience tuples
            - indices: Tree indices for updating priorities
            - importance_weights: Importance sampling weights for bias correction
        """
        batch = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Divide priority range into batch_size segments
        priority_segment = self.tree.total() / batch_size
        
        # Update beta for importance sampling
        self.beta = self._get_beta()
        self.frame += 1
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices[i] = idx
            priorities[i] = priority
        
        # Compute importance sampling weights
        # w_i = (N * P(i))^(-β) / max_j w_j
        sampling_probabilities = priorities / self.tree.total()
        importance_weights = (len(self.tree) * sampling_probabilities) ** (-self.beta)
        
        # Normalize by maximum weight for stability
        importance_weights /= importance_weights.max()
        
        return batch, indices, importance_weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Tree indices of experiences
            td_errors: TD-errors for computing new priorities
        """
        for idx, td_error in zip(indices, td_errors):
            # Priority = |TD-error| + ε
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, abs(td_error))
    
    def __len__(self):
        return len(self.tree)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if memory has enough experiences for training."""
        return len(self) >= min_size


def convert_to_standard_format(experiences: List[Experience]) -> List[Tuple]:
    """
    Convert Experience namedtuples to standard (state, action, reward, next_state, done) tuples.
    
    Args:
        experiences: List of Experience namedtuples
        
    Returns:
        List of standard tuples
    """
    return [(exp.state, exp.action, exp.reward, exp.next_state, exp.done) 
            for exp in experiences]
