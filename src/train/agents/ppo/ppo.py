"""
PPO Agent for Billiards

Implements Agent interface for evaluation using trained PPO policy.
"""

from typing import List, Optional

import numpy as np
import pooltool as pt
import torch

from src.utils.logger import get_logger

from ..base import ActionDict, Agent, BallsDict
from .ppo_networks import PolicyNetwork

logger = get_logger()


class PPOAgent(Agent):
    """PPO-trained billiards agent"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        deterministic: bool = True,
        state_dim: int = 76,
        action_dim: int = 5,
        hidden_sizes: Optional[List[int]] = None,
    ):
        """
        Initialize PPO Agent

        Args:
            model_path: Path to trained model checkpoint (optional for random initialization)
            device: Device to run on ('cuda' or 'cpu', auto-detect if None)
            deterministic: If True, use mean actions; if False, sample from distribution
            state_dim: State vector dimension
            action_dim: Action vector dimension
            hidden_sizes: Network hidden layer sizes
        """
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.deterministic = deterministic

        # Load network
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        self.policy_net = PolicyNetwork(
            state_dim=state_dim, action_dim=action_dim, hidden_sizes=hidden_sizes
        ).to(self.device)

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Support both full checkpoint and policy-only checkpoint
            if "policy_state_dict" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            elif "policy" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["policy"])
            else:
                raise ValueError(
                    "Invalid checkpoint format. Expected 'policy_state_dict' or 'policy' key."
                )
            logger.info(f"Loaded PPO model from {model_path}")

        self.policy_net.eval()
        logger.info(f"PPOAgent initialized (device={self.device}, deterministic={deterministic})")

    def decision(
        self,
        balls: Optional[BallsDict] = None,
        my_targets: Optional[List[str]] = None,
        table: Optional[pt.Table] = None,
    ) -> ActionDict:
        """
        Make a decision based on current game state

        Args:
            balls: Dict of ball objects
            my_targets: List of target ball IDs
            table: Table object

        Returns:
            ActionDict with keys: V0, phi, theta, a, b
        """
        if balls is None or my_targets is None or table is None:
            logger.warning("PPOAgent: Missing state information, returning random action")
            return self._random_action()

        try:
            # Encode state
            state = self._encode_state(balls, my_targets, table)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Get action
            with torch.no_grad():
                if self.deterministic:
                    action_raw = self.policy_net.get_action_deterministic(state_tensor)
                else:
                    action_raw, _, _ = self.policy_net.sample_action(state_tensor)

            # Map to valid range
            action = self._map_action(action_raw.squeeze(0))
            return action

        except Exception as e:
            logger.error(f"PPOAgent decision error: {e}", exc_info=True)
            return self._random_action()

    def _encode_state(self, balls: BallsDict, my_targets: List[str], table: pt.Table) -> np.ndarray:
        """
        Convert game state to 76D normalized feature vector

        State encoding:
        [0:2]    Cue ball position (x, y)
        [2:47]   15 balls × (x, y, is_pocketed)
        [47:59]  6 pockets × (x, y)
        [59:74]  Target mask (15 binary indicators)
        [74:76]  Global features (remaining_ratio, is_black8_phase)

        Args:
            balls: Dict of ball objects
            my_targets: List of target ball IDs
            table: Table object

        Returns:
            76D numpy array with normalized features
        """
        features = []

        # 1. Cue ball position (2D)
        cue_pos = balls["cue"].state.rvw[0][:2]
        features.extend([cue_pos[0] / table.l, cue_pos[1] / table.w])  # Normalize to [0, 1]

        # 2. All balls 1-15 (45D = 15 × 3)
        for i in range(1, 16):
            bid = str(i)
            ball = balls[bid]
            if ball.state.s == 4:  # Pocketed
                features.extend([0.0, 0.0, 1.0])
            else:
                pos = ball.state.rvw[0][:2]
                features.extend([pos[0] / table.l, pos[1] / table.w, 0.0])

        # 3. Pockets (12D = 6 × 2)
        for pocket_id in ["lb", "lc", "lt", "rb", "rc", "rt"]:
            pocket_pos = table.pockets[pocket_id].center[:2]
            features.extend([pocket_pos[0] / table.l, pocket_pos[1] / table.w])

        # 4. Target mask (15D)
        my_set = set(my_targets) - {"8"}  # Exclude black 8 from target set
        for i in range(1, 16):
            features.append(1.0 if str(i) in my_set else 0.0)

        # 5. Global features (2D)
        my_remaining = sum(1 for tid in my_targets if tid != "8" and balls[tid].state.s != 4)
        is_black8_phase = 1.0 if my_targets == ["8"] else 0.0
        features.extend([my_remaining / 7.0, is_black8_phase])

        return np.array(features, dtype=np.float32)

    def _map_action(self, action_raw: torch.Tensor) -> ActionDict:
        """
        Map unbounded network output to valid action space

        Network outputs are unbounded, apply activation functions:
        - V0: [0.5, 8.0] → Sigmoid
        - phi: [0, 360] → Sigmoid
        - theta: [0, 90] → Sigmoid
        - a, b: [-0.5, 0.5] → Tanh

        Args:
            action_raw: (5,) raw action tensor from network

        Returns:
            ActionDict with valid action values
        """
        V0 = float(torch.sigmoid(action_raw[0]).item() * 7.5 + 0.5)
        phi = float(torch.sigmoid(action_raw[1]).item() * 360)
        theta = float(torch.sigmoid(action_raw[2]).item() * 90)
        a = float(torch.tanh(action_raw[3]).item() * 0.5)
        b = float(torch.tanh(action_raw[4]).item() * 0.5)

        return {"V0": V0, "phi": phi, "theta": theta, "a": a, "b": b}

    def set_deterministic(self, deterministic: bool):
        """Toggle between deterministic and stochastic action selection"""
        self.deterministic = deterministic
        logger.info(f"PPOAgent deterministic mode: {deterministic}")

    def save_model(self, save_path: str):
        """Save policy network"""
        torch.save({"policy_state_dict": self.policy_net.state_dict()}, save_path)
        logger.info(f"Saved PPO model to {save_path}")
