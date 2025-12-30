"""
train/agents - Agent Package

Exports:

Agents:
- Agent: Abstract base class
- BasicAgent: Bayesian optimization-based agent, provided by TA
- BasicAgentPro: MCTS-based advanced agent, provided by TA
- RandomAgent: Random action agent
- NewAgent: Geometry-guided agent
- PPOAgent: PPO reinforcement learning agent

Types:
- ActionDict, BallsDict: Type aliases

Helper Functions: (TODO: I am not sure if these need to be exported or not)
- simulate_with_timeout: Safe simulation with timeout
- analyze_shot_for_reward: Reward calculation function
"""

from .base import (
    ActionDict,
    Agent,
    BallsDict,
    analyze_shot_for_reward,
    simulate_with_timeout,
)
from .basic import BasicAgent
from .new import NewAgent
from .ppo import PPOAgent
from .pro import BasicAgentPro
from .random import RandomAgent

__all__ = [
    "Agent",
    "ActionDict",
    "BallsDict",
    "simulate_with_timeout",
    "analyze_shot_for_reward",
    "BasicAgent",
    "BasicAgentPro",
    "RandomAgent",
    "NewAgent",
    "PPOAgent",
]
