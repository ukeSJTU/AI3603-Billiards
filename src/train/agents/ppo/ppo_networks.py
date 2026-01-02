"""
PPO Neural Networks for Billiards Agent

Implements PolicyNetwork (Actor) and ValueNetwork (Critic) for PPO training.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Actor: state → (μ, σ) for action distribution

    Architecture:
    Input(76) → Linear(256) → LayerNorm → ReLU → Dropout(0.1) →
    Linear(256) → LayerNorm → ReLU → Dropout(0.1) →
    Linear(128) → LayerNorm → ReLU → Dropout(0.1) →
    ├─ mean_head → Linear(5)
    └─ log_std_head → Linear(5) → exp → std
    """

    def __init__(self, state_dim: int = 76, action_dim: int = 5, hidden_sizes: list | None = None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Feature extraction layers
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),  # Better than BatchNorm for RL
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_size = hidden_size

        self.feature_net = nn.Sequential(*layers)

        # Separate heads for mean and std
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std_head = nn.Linear(prev_size, action_dim)

        # Orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Initialize log_std head to produce small initial std (important!)
        # Start with std ≈ 0.5 (log_std ≈ -0.7)
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, -0.7)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network

        Args:
            state: (batch, 76) state tensor

        Returns:
            mean: (batch, 5) action means
            std: (batch, 5) action standard deviations
        """
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp for numerical stability and reasonable exploration
        # Range [-2, 1] gives std in [0.135, 2.72]
        log_std = torch.clamp(log_std, min=-2.0, max=1.0)
        std = torch.exp(log_std)

        return mean, std

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with log_prob and entropy (for training)

        Args:
            state: (batch, 76) state tensor

        Returns:
            action_raw: (batch, 5) sampled actions
            log_prob: (batch,) log probabilities
            entropy: (batch,) entropies
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        action_raw = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action_raw, log_prob, entropy

    def get_action_deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation

        Args:
            state: (batch, 76) state tensor

        Returns:
            action: (batch, 5) deterministic actions (means)
        """
        mean, _ = self.forward(state)
        return mean


class ValueNetwork(nn.Module):
    """
    Critic: state → V(s)

    Architecture:
    Input(76) → Linear(256) → LayerNorm → ReLU → Dropout(0.1) →
    Linear(256) → LayerNorm → ReLU → Dropout(0.1) →
    Linear(128) → LayerNorm → ReLU → Dropout(0.1) →
    Linear(1) → squeeze → V(s)
    """

    def __init__(self, state_dim: int = 76, hidden_sizes: list | None = None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        self.state_dim = state_dim

        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network

        Args:
            state: (batch, 76) state tensor

        Returns:
            value: (batch,) state values
        """
        return self.net(state).squeeze(-1)
