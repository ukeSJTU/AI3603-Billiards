"""
config.py - 训练配置常量

设计原则：
1. 关键参数（可能需要调优或外部控制）: 通过 argparse 暴露
2. 非关键参数（不常改动）: 硬编码在此文件中

使用 dataclass 提供类型安全和默认值。
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ============ 动作空间边界（全局常量）============
@dataclass(frozen=True)
class ActionBounds:
    """动作参数边界（不可变）

    Attributes:
        V0: 初速度范围 (min, max) m/s
        phi: 水平角度范围 (min, max) 度
        theta: 垂直角度范围 (min, max) 度
        a: 杆头横向偏移范围 (min, max) 球半径比例
        b: 杆头纵向偏移范围 (min, max) 球半径比例
    """

    V0: Tuple[float, float] = (0.5, 8.0)
    phi: Tuple[float, float] = (0.0, 360.0)
    theta: Tuple[float, float] = (0.0, 90.0)
    a: Tuple[float, float] = (-0.5, 0.5)
    b: Tuple[float, float] = (-0.5, 0.5)

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        """转换为字典格式（兼容贝叶斯优化器 pbounds）"""
        return {
            "V0": self.V0,
            "phi": self.phi,
            "theta": self.theta,
            "a": self.a,
            "b": self.b,
        }


# 全局动作边界实例
ACTION_BOUNDS = ActionBounds()


# ============ 贝叶斯优化参数 ============
@dataclass
class BayesianOptConfig:
    """贝叶斯优化配置

    Attributes:
        initial_search: 初始随机采样点数
        opt_search: 后续优化迭代次数
        alpha: GP 噪声参数
        gamma_osc: 域收缩振荡系数
        gamma_pan: 域收缩平移系数
        n_restarts_optimizer: GP 优化器重启次数
        matern_nu: Matern 核参数
    """

    initial_search: int = 20
    opt_search: int = 10
    alpha: float = 1e-2
    gamma_osc: float = 0.8
    gamma_pan: float = 1.0
    n_restarts_optimizer: int = 10
    matern_nu: float = 2.5


# ============ 模拟噪声参数 ============
@dataclass
class NoiseConfig:
    """模拟噪声配置

    Attributes:
        enabled: 是否启用噪声
        V0_std: 初速度噪声标准差
        phi_std: 水平角度噪声标准差
        theta_std: 垂直角度噪声标准差
        a_std: 横向偏移噪声标准差
        b_std: 纵向偏移噪声标准差
    """

    enabled: bool = False
    V0_std: float = 0.1
    phi_std: float = 0.1
    theta_std: float = 0.1
    a_std: float = 0.003
    b_std: float = 0.003

    def as_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "V0": self.V0_std,
            "phi": self.phi_std,
            "theta": self.theta_std,
            "a": self.a_std,
            "b": self.b_std,
        }


# ============ 评估/训练配置 ============
@dataclass
class EvaluateConfig:
    """评估配置 - 这些参数可通过命令行覆盖

    Attributes:
        n_games: 对战局数
        random_seed_enabled: 是否启用固定随机种子
        random_seed: 随机种子值
        simulation_timeout: 物理模拟超时（秒）
        min_score_threshold: 低于此分数使用随机动作
    """

    n_games: int = 120
    random_seed_enabled: bool = False
    random_seed: int = 42
    simulation_timeout: int = 3
    min_score_threshold: float = 10.0


# ============ 奖励分数 ============
@dataclass(frozen=True)
class RewardConfig:
    """奖励分数配置（不可变）

    Attributes:
        own_ball_pocketed: 己方球进袋得分（每球）
        legal_eight_ball: 合法打进黑8得分
        legal_no_pocket: 合法无进球小奖励
        cue_pocketed: 白球进袋惩罚
        illegal_eight: 非法黑8惩罚
        cue_and_eight: 白球+黑8同时进袋惩罚
        foul_first_hit: 首球犯规惩罚
        foul_no_rail: 未碰库犯规惩罚
        enemy_pocketed: 对方球进袋惩罚（每球）
        simulation_failed: 模拟失败惩罚
    """

    own_ball_pocketed: int = 50
    legal_eight_ball: int = 100
    legal_no_pocket: int = 10
    cue_pocketed: int = -100
    illegal_eight: int = -150
    cue_and_eight: int = -150
    foul_first_hit: int = -30
    foul_no_rail: int = -30
    enemy_pocketed: int = -20
    simulation_failed: int = -500


# ============ 默认配置实例 ============
DEFAULT_BAYES_CONFIG = BayesianOptConfig()
DEFAULT_NOISE_CONFIG = NoiseConfig()
DEFAULT_EVAL_CONFIG = EvaluateConfig()
REWARD_CONFIG = RewardConfig()


# ============ 导出 ============
__all__ = [
    "ActionBounds",
    "ACTION_BOUNDS",
    "BayesianOptConfig",
    "DEFAULT_BAYES_CONFIG",
    "NoiseConfig",
    "DEFAULT_NOISE_CONFIG",
    "EvaluateConfig",
    "DEFAULT_EVAL_CONFIG",
    "RewardConfig",
    "REWARD_CONFIG",
]
