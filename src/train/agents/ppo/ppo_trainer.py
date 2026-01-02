"""
PPO Training Infrastructure for Billiards Agent

Implements:
- ExperienceBuffer: Stores trajectories
- HybridRewardScheduler: Manages dense→sparse reward transition
- SelfPlayTrainer: Orchestrates self-play training with PPO updates
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

from .ppo_networks import PolicyNetwork, ValueNetwork

logger = get_logger()


class ExperienceDict(TypedDict):
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: npt.NDArray[np.float32]
    values: npt.NDArray[np.float32]
    dones: npt.NDArray[np.float32]


class ExperienceBuffer:
    """Stores trajectories for PPO updates"""

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.clear()

    def clear(self):
        """Reset buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: float,
        done: bool,
    ):
        """Add a single transition"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self) -> ExperienceDict:
        """
        Get all experiences as batched tensors

        Returns:
            Dict with keys: states, actions, log_probs, rewards, values, dones
        """
        return {
            "states": torch.stack(self.states) if self.states else torch.empty(0),
            "actions": torch.stack(self.actions) if self.actions else torch.empty(0),
            "log_probs": torch.stack(self.log_probs) if self.log_probs else torch.empty(0),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }

    def __len__(self):
        return len(self.states)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation

    Formula: A_t = Σ[(γλ)^k * δ_{t+k}]
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Args:
        rewards: (T,) episode rewards
        values: (T+1,) state values (with bootstrap value at end)
        dones: (T,) done flags
        gamma: discount factor
        gae_lambda: GAE parameter

    Returns:
        advantages: (T,) advantage estimates
        returns: (T,) value targets
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)

    gae = 0.0
    for t in reversed(range(T)):
        # TD error
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]

        # GAE accumulation
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


class HybridRewardScheduler:
    """Transition from dense → sparse rewards during training"""

    def __init__(self, decay_type: str = "exponential", total_iterations: int = 1000):
        self.decay_type = decay_type
        self.total_iterations = total_iterations

    def get_dense_weight(self, iteration: int) -> float:
        """
        Weight for dense component (1.0 → 0.0)

        Args:
            iteration: Current training iteration

        Returns:
            Weight for dense reward (0.0 to 1.0)
        """
        if self.decay_type == "exponential":
            # w = exp(-5 * t/T)
            decay_rate = 5.0
            return float(np.exp(-decay_rate * iteration / self.total_iterations))

        elif self.decay_type == "linear":
            return max(0.0, 1.0 - iteration / self.total_iterations)

        elif self.decay_type == "step":
            if iteration < self.total_iterations * 0.3:
                return 1.0
            elif iteration < self.total_iterations * 0.7:
                return 0.5
            else:
                return 0.0

        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

    def compute_reward(
        self, shot_result: Dict, game_result: Optional[Dict], iteration: int, player: str
    ) -> float:
        """
        Compute hybrid reward

        Args:
            shot_result: Result from env.take_shot()
            game_result: Result from env.get_done() (None if not done)
            iteration: Current training iteration
            player: Current player ('A' or 'B')

        Returns:
            Combined reward value
        """
        dense_weight = self.get_dense_weight(iteration)
        sparse_weight = 1.0 - dense_weight

        dense_reward = self._dense_reward(shot_result)
        sparse_reward = self._sparse_reward(game_result, player) if game_result else 0.0

        return dense_weight * dense_reward + sparse_weight * sparse_reward

    def _dense_reward(self, shot_result: Dict) -> float:
        """
        Step reward based on analyze_shot_for_reward

        Normalized to [-1, 1]
        """
        reward = 0.0

        # Own balls pocketed
        reward += len(shot_result.get("ME_INTO_POCKET", [])) * 50

        # Enemy balls pocketed (penalty)
        reward -= len(shot_result.get("ENEMY_INTO_POCKET", [])) * 20

        # Cue ball pocketed
        if shot_result.get("WHITE_BALL_INTO_POCKET", False):
            reward -= 100

        # Black 8 handling
        if shot_result.get("BLACK_BALL_INTO_POCKET", False):
            # Legal black 8 would have ended the game (handled in sparse)
            # This is illegal black 8
            reward -= 150

        # Fouls
        if shot_result.get("FOUL_FIRST_HIT", False):
            reward -= 30
        if shot_result.get("NO_POCKET_NO_RAIL", False):
            reward -= 30

        # Legal shot bonus
        if reward == 0:
            reward = 10

        # Normalize to [-1, 1]
        return float(np.clip(reward / 150.0, -1.0, 1.0))

    def _sparse_reward(self, game_result: Dict, player: str) -> float:
        """
        Terminal reward: win=+1, loss=-1, draw=0

        Args:
            game_result: Dict with 'winner' key
            player: Current player ('A' or 'B')

        Returns:
            Sparse reward
        """
        winner = game_result.get("winner")

        if winner == player:
            return 1.0
        elif winner == "SAME":
            return 0.0
        else:
            return -1.0


def ppo_loss(
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    policy_net: PolicyNetwork,
    value_net: ValueNetwork,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PPO clipped objective + value loss + entropy bonus

    L = L_CLIP(θ) + c1*L_VF(θ) - c2*H(π_θ)

    Args:
        states: (batch, 76) state tensors
        actions: (batch, 5) action tensors
        old_log_probs: (batch,) old log probabilities
        advantages: (batch,) advantage estimates
        returns: (batch,) value targets
        policy_net: Policy network
        value_net: Value network
        clip_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient

    Returns:
        loss: Combined loss
        metrics: Dict with loss components and diagnostics
    """
    # Policy loss
    mean, std = policy_net(states)
    dist = torch.distributions.Normal(mean, std)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    values = value_net(states)
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus
    entropy = dist.entropy().sum(dim=-1).mean()

    # Combined loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # Metrics for logging
    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean().item()
        clipfrac = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
    }

    return loss, metrics


class SelfPlayTrainer:
    """Manages self-play training with opponent snapshots"""

    def __init__(self, config: Dict, checkpoint_dir: str = "checkpoints/ppo"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        network_config = config.get("network", {})
        self.policy_net = PolicyNetwork(
            state_dim=network_config.get("state_dim", 76),
            action_dim=network_config.get("action_dim", 5),
            hidden_sizes=network_config.get("hidden_sizes", [256, 256, 128]),
        ).to(self.device)

        self.value_net = ValueNetwork(
            state_dim=network_config.get("state_dim", 76),
            hidden_sizes=network_config.get("hidden_sizes", [256, 256, 128]),
        ).to(self.device)

        # Optimizers
        training_config = config.get("training", {})
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=training_config.get("learning_rate_policy", 3e-4)
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=training_config.get("learning_rate_value", 1e-3)
        )

        # Hyperparameters
        self.gamma = training_config.get("gamma", 0.99)
        self.gae_lambda = training_config.get("gae_lambda", 0.95)
        self.clip_epsilon = training_config.get("clip_epsilon", 0.2)
        self.n_epochs = training_config.get("n_epochs", 10)
        self.batch_size = training_config.get("batch_size", 64)
        self.max_grad_norm = training_config.get("max_grad_norm", 0.5)

        # Experience buffer
        buffer_size = training_config.get("buffer_size", 2048)
        self.buffer = ExperienceBuffer(capacity=buffer_size)

        # Reward scheduler
        reward_config = config.get("reward", {})
        self.reward_scheduler = HybridRewardScheduler(
            decay_type=reward_config.get("decay_type", "exponential"),
            total_iterations=reward_config.get("total_iterations", 1000),
        )

        # Opponent pool
        self_play_config = config.get("self_play", {})
        self.opponent_pool: List[Path] = []
        self.max_pool_size = self_play_config.get("max_opponent_pool_size", 5)
        self.snapshot_interval = self_play_config.get("snapshot_interval", 50)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SelfPlayTrainer initialized (device={self.device})")
        logger.info(f"Policy params: {sum(p.numel() for p in self.policy_net.parameters())}")
        logger.info(f"Value params: {sum(p.numel() for p in self.value_net.parameters())}")

    def save_opponent_snapshot(self, iteration: int):
        """Save current policy as opponent"""
        snapshot_path = self.checkpoint_dir / f"opponent_{iteration}.pth"
        torch.save({"policy": self.policy_net.state_dict(), "iteration": iteration}, snapshot_path)

        self.opponent_pool.append(snapshot_path)

        # Keep only recent snapshots
        if len(self.opponent_pool) > self.max_pool_size:
            old_path = self.opponent_pool.pop(0)
            if old_path.exists():
                old_path.unlink()

        logger.info(f"Saved opponent snapshot at iteration {iteration}")

    def load_random_opponent(self) -> Optional[PolicyNetwork]:
        """Load random opponent from pool"""
        if not self.opponent_pool:
            return None  # Early training: no opponents yet

        opponent_path = random.choice(self.opponent_pool)
        checkpoint = torch.load(opponent_path, map_location=self.device)

        opponent = PolicyNetwork(
            state_dim=self.config["network"].get("state_dim", 76),
            action_dim=self.config["network"].get("action_dim", 5),
            hidden_sizes=self.config["network"].get("hidden_sizes", [256, 256, 128]),
        ).to(self.device)

        opponent.load_state_dict(checkpoint["policy"])
        opponent.eval()

        logger.info(f"Loaded opponent from {opponent_path.name}")
        return opponent

    def update_ppo(self) -> Dict[str, float]:
        """
        Perform PPO update using buffered experiences

        Returns:
            Metrics dict with averaged loss components
        """
        # Get experiences
        data = self.buffer.get()
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        rewards = data["rewards"]
        dones = data["dones"]

        # Compute values for GAE
        with torch.no_grad():
            values = self.value_net(states).cpu().numpy()
            # Bootstrap value: 0 if episode ended, otherwise V(last_state)
            if len(dones) > 0 and dones[-1]:
                # Episode terminated, bootstrap value is 0
                bootstrap_value = 0.0
            else:
                # Episode not finished, use last state value
                bootstrap_value = values[-1] if len(values) > 0 else 0.0
            values = np.append(values, bootstrap_value)

        # Compute advantages and returns
        advantages, returns = compute_gae(
            rewards, values, dones, gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        total_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }

        n_batches = 0
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))

            # Mini-batch updates
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Compute loss
                value_coef = self.config.get("training", {}).get("value_coef", 0.5)
                entropy_coef = self.config.get("training", {}).get("entropy_coef", 0.01)

                loss, metrics = ppo_loss(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                    self.policy_net,
                    self.value_net,
                    self.clip_epsilon,
                    value_coef,
                    entropy_coef,
                )

                # Update policy
                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                # Update value (using same loss, but separate optimizer)
                self.value_optimizer.zero_grad()
                values_pred = self.value_net(batch_states)
                value_loss = F.mse_loss(values_pred, batch_returns)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                # Accumulate metrics
                for key in total_metrics:
                    total_metrics[key] += metrics.get(key, 0.0)
                n_batches += 1

        # Average metrics
        avg_metrics = {key: val / n_batches for key, val in total_metrics.items()}

        return avg_metrics

    def save_checkpoint(self, iteration: int, metrics: Dict):
        """Save full training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}.pth"
        torch.save(
            {
                "iteration": iteration,
                "policy_state_dict": self.policy_net.state_dict(),
                "value_state_dict": self.value_net.state_dict(),
                "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
                "value_optimizer_state_dict": self.value_optimizer.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )
        logger.info(f"Saved checkpoint at iteration {iteration}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])

        iteration = checkpoint["iteration"]
        logger.info(f"Loaded checkpoint from iteration {iteration}")
        return iteration
