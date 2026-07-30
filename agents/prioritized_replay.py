"""Prioritized replay adapted from HEART-CH's Rainbow infrastructure."""

from __future__ import annotations

import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.data = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, transition):
        priority = (
            float(self.priorities[: self.size].max())
            if self.size
            else 1.0
        )
        self.data[self.position] = transition
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        priorities = self.priorities[: self.size]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(
            self.size, size=batch_size, replace=False, p=probabilities
        )
        weights = (self.size * probabilities[indices]) ** (-float(beta))
        weights /= weights.max()
        return (
            [self.data[index] for index in indices],
            indices,
            weights.astype(np.float32),
        )

    def update(self, indices, errors):
        self.priorities[np.asarray(indices)] = (
            np.abs(np.asarray(errors, dtype=np.float64)) + 1e-6
        )
